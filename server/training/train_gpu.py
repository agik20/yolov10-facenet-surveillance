import os
import json
import torch
import cupy as cp
import cudf

from facenet_pytorch import InceptionResnetV1
from joblib import dump

from preprocessing import load_faces  # pastikan mengembalikan tensor [C,H,W] ter-normalisasi [0,1], RGB

# =====================
# RAPIDS / cuML (GPU)
# =====================
from cuml.preprocessing import Normalizer, LabelEncoder
from cuml.decomposition import PCA
from cuml.svm import SVC
from cuml.model_selection import train_test_split

# =====================
# KONFIGURASI
# =====================
data_dir = 'augmented_faces/train'
use_pca = True
pca_components = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

normalizer = Normalizer(norm='l2')  # cuML Normalizer (GPU)

os.makedirs("models", exist_ok=True)

# =====================
# UTIL GPU
# =====================
def torch_tensor_to_cupy(t: torch.Tensor) -> cp.ndarray:
    """
    Konversi torch CUDA tensor -> CuPy via DLPack (tanpa copy ke host).
    """
    assert t.is_cuda, "Tensor harus di GPU (CUDA) sebelum konversi ke CuPy."
    from torch.utils.dlpack import to_dlpack
    return cp.fromDlpack(to_dlpack(t))

# =====================
# FUNGSI
# =====================
def extract_embeddings():
    """
    Ekstrak embedding FaceNet di GPU.
    Output:
      X: CuPy array [N, 512] (atau dim FaceNet)
      y: cuDF Series (string labels)
    """
    emb_list = []
    labels = []

    for class_name in os.listdir(data_dir):
        folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder):
            continue
        print(f"📂 Memproses kelas: {class_name}")
        faces = load_faces(folder)  # iterable of torch tensors [C,H,W], sudah di-normalisasi untuk Facenet
        for face in faces:
            with torch.no_grad():
                # Pastikan tensor di GPU dan bentuk [1,3,H,W]
                if not face.is_cuda:
                    face = face.to(device)
                if face.dim() == 3:
                    face = face.unsqueeze(0)

                emb_torch = facenet(face)  # [1, 512] di GPU
                emb_cu = torch_tensor_to_cupy(emb_torch.squeeze(0))  # [512]
                emb_list.append(emb_cu)
                labels.append(class_name)

    # Stack ke CuPy [N, D]
    X = cp.stack(emb_list, axis=0) if len(emb_list) > 0 else cp.empty((0, 512), dtype=cp.float32)
    y = cudf.Series(labels)  # GPU string column
    return X, y

def compute_confusion_matrix_gpu(y_true_cp: cp.ndarray, y_pred_cp: cp.ndarray, n_classes: int) -> cp.ndarray:
    """
    Confusion matrix murni di GPU (CuPy).
    """
    flat = y_true_cp.astype(cp.int64) * n_classes + y_pred_cp.astype(cp.int64)
    counts = cp.bincount(flat, minlength=n_classes * n_classes)
    cm = counts.reshape((n_classes, n_classes))
    return cm

def per_class_metrics_gpu(cm: cp.ndarray):
    """
    Hitung precision, recall, f1 per kelas di GPU.
    """
    eps = cp.finfo(cp.float32).eps
    tp = cp.diag(cm).astype(cp.float32)
    fp = cm.sum(axis=0).astype(cp.float32) - tp
    fn = cm.sum(axis=1).astype(cp.float32) - tp
    support = cm.sum(axis=1).astype(cp.int32)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1, support

def train_svm():
    print("\n📊 Memulai ekstraksi embedding (GPU end-to-end)...")
    X, y = extract_embeddings()
    n_samples = int(X.shape[0])
    print(f"Total data embedding: {n_samples}")

    if n_samples == 0:
        raise RuntimeError("Tidak ada data ditemukan pada 'augmented_faces/train'.")

    print("⚡ Normalisasi di GPU (cuML)...")
    X = normalizer.fit_transform(X)  # CuPy in, CuPy out

    if use_pca:
        print("📉 Menerapkan PCA (GPU, cuML)...")
        pca = PCA(n_components=pca_components, whiten=False)
        X = pca.fit_transform(X)  # CuPy
        dump(pca, 'models/pca_model.pkl')
        print(f"Dimensi setelah PCA: {tuple(map(int, X.shape))}")

    print("🔖 Encoding label (GPU, cuML)...")
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)  # y_enc: cudf.Series of int32
    dump(encoder, 'models/in_encoder.pkl')
    
    # Ambil kelas ke host
    classes = encoder.classes_.to_pandas().tolist()
    n_classes = len(classes)

    print("🔀 Train/Test split (GPU, cuML)...")
    # train_test_split dari cuML mendukung stratify (cudf.Series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=42, shuffle=True
    )
    print(f"🧠 Jumlah data pelatihan: {int(X_train.shape[0])}, pengujian: {int(X_test.shape[0])}")

    print("🤖 Melatih model SVM (GPU, cuML)...")
    model = SVC(kernel='linear')  # tanpa probability=True (belum tersedia di cuML)
    model.fit(X_train, y_train)
    dump(model, 'models/svm_model.pkl')
    print("💾 Model SVM berhasil disimpan (GPU).")

    print("\n📈 Evaluasi (GPU murni dengan CuPy/cuDF)...")
    # y_test dan y_pred sebagai CuPy arrays (int32)
    # y_test (cudf.Series) -> CuPy
    y_test_cp = y_test.to_cupy()
    y_pred_cp = model.predict(X_test)  # CuPy

    # Akurasi
    acc = cp.mean((y_pred_cp.astype(cp.int64) == y_test_cp.astype(cp.int64)).astype(cp.float32))
    accuracy = float(acc.get())  # ambil nilai ke host untuk print/log

    # Confusion Matrix di GPU
    cm_cp = compute_confusion_matrix_gpu(y_test_cp, y_pred_cp, n_classes=n_classes)

    # Per-class metrics di GPU
    precision_cp, recall_cp, f1_cp, support_cp = per_class_metrics_gpu(cm_cp)

    # Macro & weighted (GPU)
    support_float = support_cp.astype(cp.float32)
    total = support_float.sum()
    macro_precision = float(precision_cp.mean().get())
    macro_recall = float(recall_cp.mean().get())
    macro_f1 = float(f1_cp.mean().get())

    weights = support_float / (total + cp.finfo(cp.float32).eps)
    weighted_precision = float((precision_cp * weights).sum().get())
    weighted_recall = float((recall_cp * weights).sum().get())
    weighted_f1 = float((f1_cp * weights).sum().get())

    # Siapkan laporan ringkas (konversi minimal ke host untuk simpan JSON)
    metrics_report = {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1
        },
        "per_class": {
            cls_name: {
                "precision": float(precision_cp[i].get()),
                "recall": float(recall_cp[i].get()),
                "f1": float(f1_cp[i].get()),
                "support": int(support_cp[i].get()),
            }
            for i, cls_name in enumerate(classes)
        }
    }

    # Simpan confusion matrix (ke CSV) & report (ke JSON)
    cm_host = cm_cp.get()  # transfer sekali untuk disimpan
    cm_path = "models/confusion_matrix.csv"
    with open(cm_path, "w") as f:
        # header
        f.write("," + ",".join(map(str, classes)) + "\n")
        for i, row in enumerate(cm_host):
            f.write(f"{classes[i]}," + ",".join(map(str, row)) + "\n")

    report_path = "models/metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    print(f"📄 Metrics JSON: {report_path}")
    print(f"🧭 Confusion Matrix CSV: {cm_path}")

if __name__ == '__main__':
    train_svm()