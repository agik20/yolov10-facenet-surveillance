// script.js
document.addEventListener("DOMContentLoaded", () => {
  // === CCTV STREAM ===
  const liveVideo = document.getElementById("live_video");
  if (liveVideo) {
    // Ganti IP sesuai alamat Raspberry Pi Anda
    liveVideo.src = "http://pi:5000/video";
  }

  // === FORM SEARCH (untuk search.php / history.php) ===
  const form = document.querySelector(".search-box form");
  if (form) {
    form.addEventListener("submit", (e) => {
      const input = form.querySelector("input[name='nik']");
      const nik = input.value.trim();

      // Validasi input kosong
      if (nik === "") {
        e.preventDefault();
        alert("Mohon masukkan NIK terlebih dahulu!");
        return;
      }

      // Jika ingin AJAX (tanpa reload)
      e.preventDefault();
      const params = new URLSearchParams(new FormData(form));

      // Tentukan target container (search / history)
      let targetBox =
        document.querySelector(".result-box") ||
        document.querySelector(".history-container");
      if (!targetBox) return;

      // Tampilkan indikator loading
      targetBox.innerHTML = "<p>Sedang mencari data...</p>";

      // Fetch data dari server
      fetch(window.location.pathname + "?" + params.toString(), {
        headers: {
          "X-Requested-With": "XMLHttpRequest",
        },
      })
        .then((res) => res.text())
        .then((html) => {
          // Buat elemen dummy utk parsing
          const parser = new DOMParser();
          const doc = parser.parseFromString(html, "text/html");

          // Ambil hanya bagian hasil pencarian
          const newContent =
            doc.querySelector(".result-box") ||
            doc.querySelector(".history-container");

          if (newContent) {
            targetBox.innerHTML = newContent.innerHTML;
          } else {
            targetBox.innerHTML =
              "<p>Terjadi kesalahan saat mengambil data.</p>";
          }
        })
        .catch((err) => {
          console.error(err);
          targetBox.innerHTML = "<p>Gagal mengambil data.</p>";
        });
    });
  }
});
