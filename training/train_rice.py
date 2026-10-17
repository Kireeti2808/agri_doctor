"""
Train, evaluate and export the Agri-Doctor Rice disease model.

Run on a machine with a GPU (Colab works):

    python train_rice.py --data /path/to/rice_dataset --out rice_run

The dataset folder needs one sub-folder per class, named exactly like the
app's RICE_CLASSES (see agri_core.py); images are split here, so do not
pre-split. Outputs in --out:

    rice_model.tflite        float32, input [1, 224, 224, 3] raw 0-255 RGB
    labels.txt               class names in output order
    metrics.json/.md         per-class precision/recall/F1, confusion matrix
    data_report.json         class counts, duplicates removed, split sizes
    split.json               which file went to which split (reproducible)

Nothing in this script has been run on the real Rice dataset yet; only a
synthetic smoke test (tests/test_training.py) proves the mechanics.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agri_core import RICE_CLASSES  # noqa: E402

IMAGE_SIZE = 224
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The exported labels.txt used "Rice_Healthy_Rice_Leaf"; accept either
# folder name for the healthy class.
FOLDER_ALIASES = {"Rice_Healthy_Rice_Leaf": "Rice_Healthy"}


# ============================================================
# DATA AUDIT AND LEAKAGE-SAFE SPLIT
# ============================================================

def average_hash(path, size=8):
    """64-bit perceptual hash: identical for exact and near-exact copies."""
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32
        )
    bits = (pixels > pixels.mean()).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def collect_images(data_dir):
    found, unknown, unreadable = [], [], []

    for folder in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        label = FOLDER_ALIASES.get(folder.name, folder.name)

        if label not in RICE_CLASSES:
            unknown.append(folder.name)
            continue

        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                found.append((path, label))
            except Exception:
                unreadable.append(str(path))

    return found, unknown, unreadable


def group_near_duplicates(items, threshold):
    """
    Union images whose hashes differ by <= threshold bits, so copies and
    near-copies (crops, re-encodes, augmentations saved to disk) always land
    in the same split. O(n^2) on hashes; fine for tens of thousands.
    """
    hashes = [average_hash(path) for path, _ in items]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if hamming(hashes[i], hashes[j]) <= threshold:
                parent[find(i)] = find(j)

    groups = defaultdict(list)
    for index in range(len(items)):
        groups[find(index)].append(index)

    return list(groups.values())


def split_groups(items, groups, val_fraction, test_fraction, seed):
    """
    Per-class group split: each class contributes roughly the requested
    fractions to validation and test, and no duplicate group straddles two
    splits.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)

    for group in groups:
        labels = Counter(items[index][1] for index in group)
        by_class[labels.most_common(1)[0][0]].append(group)

    splits = {"train": [], "val": [], "test": []}

    for label in sorted(by_class):
        class_groups = by_class[label]
        rng.shuffle(class_groups)
        total = sum(len(group) for group in class_groups)
        targets = {"test": total * test_fraction, "val": total * val_fraction}
        filled = {"test": 0, "val": 0}

        for group in class_groups:
            for split in ("test", "val"):
                if filled[split] < targets[split]:
                    splits[split].extend(group)
                    filled[split] += len(group)
                    break
            else:
                splits["train"].extend(group)

    return splits


def audit_and_split(args):
    items, unknown, unreadable = collect_images(args.data)

    if not items:
        raise SystemExit(f"No images found under {args.data}.")

    counts = Counter(label for _, label in items)
    missing = [label for label in RICE_CLASSES if counts[label] == 0]

    if missing:
        raise SystemExit(
            f"Missing class folders (or empty): {missing}. Expected one "
            f"folder per class: {RICE_CLASSES}"
        )

    groups = group_near_duplicates(items, args.dup_threshold)
    duplicate_images = sum(len(group) - 1 for group in groups if len(group) > 1)
    mixed_label_groups = [
        sorted({str(items[i][0]) for i in group})
        for group in groups
        if len({items[i][1] for i in group}) > 1
    ]

    splits = split_groups(items, groups, args.val, args.test, args.seed)

    largest, smallest = max(counts.values()), min(counts.values())

    report = {
        "images": len(items),
        "class_counts": {label: counts[label] for label in RICE_CLASSES},
        "imbalance_ratio": round(largest / smallest, 2),
        "duplicate_groups": sum(1 for g in groups if len(g) > 1),
        "images_in_duplicate_groups_beyond_first": duplicate_images,
        "possible_label_conflicts": mixed_label_groups[:50],
        "possible_label_conflict_count": len(mixed_label_groups),
        "unknown_folders": unknown,
        "unreadable_files": unreadable[:50],
        "split_sizes": {name: len(idx) for name, idx in splits.items()},
        "split_class_counts": {
            name: dict(Counter(items[i][1] for i in idx))
            for name, idx in splits.items()
        }
    }

    split_files = {
        name: [[str(items[i][0]), items[i][1]] for i in idx]
        for name, idx in splits.items()
    }

    return items, splits, report, split_files


# ============================================================
# TRAINING
# ============================================================

def make_dataset(tf, items, indices, batch, shuffle, seed):
    paths = [str(items[i][0]) for i in indices]
    labels = [RICE_CLASSES.index(items[i][1]) for i in indices]

    def load(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return tf.cast(image, tf.float32), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(tf, weights, dropout):
    """
    EfficientNetV2-B0 with preprocessing built in, matching the Maize and
    Cotton models: the app feeds raw 0-255 RGB at 224x224.
    """
    layers = tf.keras.layers

    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.15, value_range=(0, 255))
        ],
        name="augment"
    )

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_preprocessing=True,
        pooling="avg"
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = augment(inputs)
    x = backbone(x, training=False)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(
        len(RICE_CLASSES),
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    return tf.keras.Model(inputs, outputs, name="rice_doctor"), backbone


def class_weights(items, train_indices):
    counts = Counter(RICE_CLASSES.index(items[i][1]) for i in train_indices)
    total = sum(counts.values())
    return {
        index: total / (len(RICE_CLASSES) * counts[index])
        for index in range(len(RICE_CLASSES))
        if counts[index]
    }


def train(tf, args, items, splits):
    tf.keras.utils.set_random_seed(args.seed)

    train_ds = make_dataset(tf, items, splits["train"], args.batch, True, args.seed)
    val_ds = make_dataset(tf, items, splits["val"], args.batch, False, args.seed)

    weights = None if args.weights == "none" else args.weights
    model, backbone = build_model(tf, weights, args.dropout)
    weight_map = class_weights(items, splits["train"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=max(1, args.patience // 2)
        )
    ]

    # Phase 1: train the new head on a frozen backbone.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_head,
        class_weight=weight_map, callbacks=callbacks, verbose=2
    )

    # Phase 2: fine-tune the top of the backbone at a low learning rate.
    history_tune = None

    if args.epochs_finetune > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-args.unfreeze]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        history_tune = model.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
            class_weight=weight_map, callbacks=callbacks, verbose=2
        )

    history = {
        "head": history_head.history,
        "finetune": history_tune.history if history_tune else None
    }

    return model, history


# ============================================================
# EVALUATION
# ============================================================

def evaluate(predict, items, indices):
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred = [], []

    for index in indices:
        path, label = items[index]
        with Image.open(path) as image:
            array = np.asarray(
                image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)),
                dtype=np.float32
            )
        y_true.append(RICE_CLASSES.index(label))
        y_pred.append(int(np.argmax(predict(array))))

    labels = list(range(len(RICE_CLASSES)))

    return {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=RICE_CLASSES,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "test_images": len(indices)
    }


def tflite_predictor(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    def predict(array):
        interpreter.set_tensor(
            input_details["index"],
            array[None].astype(input_details["dtype"])
        )
        interpreter.invoke()
        return interpreter.get_tensor(output_details["index"])[0]

    return predict, interpreter


def metrics_markdown(title, metrics):
    lines = [
        f"## {title}",
        "",
        f"Test images: {metrics['test_images']} · accuracy "
        f"{metrics['accuracy'] * 100:.1f}%",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|"
    ]

    for label in RICE_CLASSES:
        row = metrics["per_class"][label]
        lines.append(
            f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1-score']:.3f} | {int(row['support'])} |"
        )

    macro = metrics["per_class"]["macro avg"]
    lines += [
        f"| macro avg | {macro['precision']:.3f} | {macro['recall']:.3f} | "
        f"{macro['f1-score']:.3f} | |",
        "",
        "Confusion matrix (rows = true, columns = predicted, order as above):",
        "",
        "```"
    ]
    lines += [" ".join(f"{v:4d}" for v in row) for row in metrics["confusion_matrix"]]
    lines += ["```", ""]

    return "\n".join(lines)


# ============================================================
# EXPORT AND VERIFICATION
# ============================================================

def export_tflite(tf, model, out_dir):
    # Drop the augmentation block: it is a no-op at inference, but keeping
    # inference graphs minimal avoids unsupported ops in the converter.
    inference = tf.keras.Model(
        model.input,
        model.layers[-1](model.layers[-2](model.layers[-3](model.input, training=False)))
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(inference)
    path = out_dir / "rice_model.tflite"
    path.write_bytes(converter.convert())
    (out_dir / "labels.txt").write_text("\n".join(RICE_CLASSES) + "\n")
    return path, inference


def verify_export(path, keras_model, items, indices):
    """
    Check the exported artifact against what the app requires and against
    the Keras model it came from.
    """
    from agri_core import check_output_classes, validate_tflite_file

    validate_tflite_file(path)
    predict, interpreter = tflite_predictor(path)
    check_output_classes(interpreter, RICE_CLASSES, "Rice")

    details = interpreter.get_input_details()[0]
    assert list(details["shape"]) == [1, IMAGE_SIZE, IMAGE_SIZE, 3], details["shape"]
    assert details["dtype"] == np.float32, details["dtype"]

    sample = indices[: min(16, len(indices))]
    max_difference, agreement = 0.0, 0

    for index in sample:
        with Image.open(items[index][0]) as image:
            array = np.asarray(
                image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32
            )
        lite = predict(array)
        full = keras_model.predict(array[None], verbose=0)[0]
        max_difference = max(max_difference, float(np.max(np.abs(lite - full))))
        agreement += int(np.argmax(lite) == np.argmax(full))
        assert abs(float(lite.sum()) - 1.0) < 1e-3, "outputs are not softmax"

    return {
        "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE, 3],
        "input_dtype": "float32",
        "input_range": "raw 0-255 RGB (rescaling is inside the model)",
        "outputs": len(RICE_CLASSES),
        "labels": RICE_CLASSES,
        "keras_vs_tflite_max_abs_difference": max_difference,
        "keras_vs_tflite_top1_agreement": f"{agreement}/{len(sample)}",
        "size_mb": round(path.stat().st_size / 1e6, 2)
    }


# ============================================================
# MAIN
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="folder with one sub-folder per class")
    parser.add_argument("--out", default="rice_run")
    parser.add_argument("--baseline", help="existing rice .tflite to evaluate on the same test split")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--dup-threshold", type=int, default=4,
                        help="max hash bit difference treated as a near-duplicate")
    parser.add_argument("--epochs-head", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=20)
    parser.add_argument("--unfreeze", type=int, default=40, help="backbone layers to fine-tune")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true", help="report data and split, do not train")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items, splits, report, split_files = audit_and_split(args)
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "split.json").write_text(json.dumps(split_files, indent=2))
    print(json.dumps({k: report[k] for k in ("images", "class_counts", "imbalance_ratio",
                                             "duplicate_groups", "possible_label_conflict_count",
                                             "split_sizes")}, indent=2))

    if args.audit_only:
        return report

    results = {"data": report}
    markdown = ["# Rice model evaluation", ""]

    if args.baseline:
        predict, interpreter = tflite_predictor(args.baseline)
        outputs = int(interpreter.get_output_details()[0]["shape"][-1])

        if outputs == len(RICE_CLASSES):
            results["baseline"] = evaluate(predict, items, splits["test"])
            markdown.append(metrics_markdown(f"Baseline: {Path(args.baseline).name}", results["baseline"]))
        else:
            results["baseline"] = f"skipped: baseline has {outputs} outputs, expected {len(RICE_CLASSES)}"

    import tensorflow as tf

    model, history = train(tf, args, items, splits)
    results["history"] = history

    model_path, inference_model = export_tflite(tf, model, out_dir)
    results["export"] = verify_export(model_path, inference_model, items, splits["test"] or splits["val"])

    predict, _ = tflite_predictor(model_path)
    results["final"] = evaluate(predict, items, splits["test"])
    markdown.append(metrics_markdown("Final model (exported TFLite)", results["final"]))

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    (out_dir / "metrics.md").write_text("\n".join(markdown))
    print("\n".join(markdown))
    print(json.dumps(results["export"], indent=2))

    return results


if __name__ == "__main__":
    main()
