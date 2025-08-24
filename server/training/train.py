import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump
from facenet_pytorch import InceptionResnetV1
from preprocessing import load_faces

data_dir = 'augmented_faces/train'
use_pca = True
pca_components = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
normalizer = Normalizer(norm='l2')

def extract_embeddings():
    embeddings, labels = [], []

    for class_name in os.listdir(data_dir):
        folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder):
            continue

        print(f"📂 Memproses kelas: {class_name}")
        faces = load_faces(folder)

        for face in faces:
            with torch.no_grad():
                face = face.to(device).unsqueeze(0)
                emb = facenet(face).cpu().numpy()[0]
                embeddings.append(emb)
                labels.append(class_name)

    return np.array(embeddings), np.array(labels)

def plot_evaluation_report(cm, class_names, report_dict, accuracy, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Ambil nilai metrik per kelas
    precisions = [report_dict[label]['precision'] for label in class_names]
    recalls = [report_dict[label]['recall'] for label in class_names]
    f1s = [report_dict[label]['f1-score'] for label in class_names]

    num_classes = len(class_names)
    fig_width = 16
    fig_height = max(9, num_classes * 0.6)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = plt.GridSpec(2, 2, width_ratios=[1.2, 1.8], height_ratios=[1, 1.2], hspace=0.4, wspace=0.35)

    # Confusion Matrix (Kiri penuh)
    ax1 = fig.add_subplot(grid[:, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, square=True, annot_kws={"size": 12}, ax=ax1)
    ax1.set_title("Confusion Matrix", fontsize=14)
    ax1.set_xlabel("Predicted", fontsize=12)
    ax1.set_ylabel("Actual", fontsize=12)

    # Bar Chart (Kanan atas)
    ax2 = fig.add_subplot(grid[0, 1])
    x = np.arange(num_classes)
    ax2.bar(x - 0.2, precisions, width=0.2, label='Precision', color='skyblue')
    ax2.bar(x, recalls, width=0.2, label='Recall', color='orange')
    ax2.bar(x + 0.2, f1s, width=0.2, label='F1 Score', color='green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Metrik per Kelas", fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Classification Report (Kanan bawah)
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.axis('off')
    lines = ["Classification Report:\n"]
    for label in class_names:
        m = report_dict[label]
        lines.append(f"{label:<10}  P: {m['precision']:.2f}  R: {m['recall']:.2f}  F1: {m['f1-score']:.2f}  N: {m['support']}")
    lines.append(f"\nAccuracy: {accuracy * 100:.2f}%")
    ax3.text(0.01, 1, '\n'.join(lines), fontsize=11, va='top', family='monospace')

    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.06)
    plt.savefig(filename)
    plt.close()

def train_svm():
    print("\n📊 Memulai ekstraksi embedding...")
    X, y = extract_embeddings()
    print(f"Total data embedding: {len(X)}")

    X = normalizer.transform(X)

    if use_pca:
        print("📉 Menerapkan PCA untuk reduksi dimensi...")
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        dump(pca, 'models/pca_model.pkl')
        print(f"Dimensi setelah PCA: {X.shape}")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    dump(encoder, 'models/in_encoder.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=42
    )
    print(f"🧠 Jumlah data pelatihan: {len(X_train)}, pengujian: {len(X_test)}")

    print("🤖 Melatih model SVM...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    dump(model, 'models/svm_model.pkl')
    print("💾 Model SVM berhasil disimpan.")

    print("\n📈 Evaluasi model sedang diproses...")
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    plot_evaluation_report(cm, encoder.classes_, report, accuracy, "models/evaluation_report.png")

    print("📊 Laporan evaluasi disimpan dalam: models/evaluation_report.png")

if __name__ == '__main__':
    train_svm()