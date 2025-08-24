<?php
// Koneksi ke database
require_once 'config.php';

// Helper: convert absolute path -> web path
function toWebPath($p) {
    if (!$p) return '';
    return str_replace('/var/www/html/', '/', $p);
}

// Ambil filter NIK (opsional)
$nik = isset($_GET['nik']) ? trim($_GET['nik']) : '';

// Query history (pakai prepared statement)
if ($nik !== '') {
    $sql = "
        SELECT plh.*, u.name, u.NIK, c.location_name
        FROM person_location_history plh
        JOIN users u ON plh.user_id = u.user_id
        JOIN cameras c ON plh.camera_id = c.camera_id
        WHERE u.NIK = ?
        ORDER BY plh.seen_at DESC
    ";
    $stmt = $mysqli->prepare($sql);
    $stmt->bind_param('s', $nik);
    $stmt->execute();
    $history = $stmt->get_result();
} else {
    $sql = "
        SELECT plh.*, u.name, u.NIK, c.location_name
        FROM person_location_history plh
        JOIN users u ON plh.user_id = u.user_id
        JOIN cameras c ON plh.camera_id = c.camera_id
        ORDER BY plh.seen_at DESC
    ";
    $history = $mysqli->query($sql);
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Gotham</title>
    <link rel="stylesheet" href="style.css">
    <script src="script.js" defer></script>
    <style>
        .container img {
            max-width: 100%;
            border-radius: 15px;
        }
    </style>
</head>
<body>
    <!-- Layer untuk animasi transisi -->
    <div class="page-transition">
        <div class="page-content">
            
            <div class="decor_kiri">
                <img src="assets/addition/kiri.png" alt="decor kiri" />
            </div>

            <header data-page="history">
                <a href="index.php">Home</a>
                <a href="cctv.php">CCTV</a>
                <a href="search.php">Search</a>
                <a href="#" class="orange">History</a>
            </header>

            <div class="container">
                <div class="search-box">
                    <form method="get" action="">
                        <input type="text" name="nik" placeholder="Search by ID (NIK)" value="<?php echo htmlspecialchars($nik); ?>">
                        <button type="submit">Search</button>
                    </form>
                </div>

                <div class="history-container">
                    <?php
                    if ($history && $history->num_rows > 0) {
                        $no = 1;
                        while ($row = $history->fetch_assoc()) {

                            // Jika difilter NIK tapi baris ini bukan NIK tsb (redundan karena SQL sudah filter), skip
                            if ($nik !== '' && $row['NIK'] !== $nik) {
                                continue;
                            }

                            $imgCrop  = toWebPath($row['image_path']);       // crop
                            $imgFull  = toWebPath($row['full_frame_path']);  // full frame (kalau mau dipakai)
                            $name     = htmlspecialchars($row['name']);
                            $NIK      = htmlspecialchars($row['NIK']);
                            $loc      = htmlspecialchars($row['location_name']);
                            $seenAt   = htmlspecialchars($row['seen_at']);
                    ?>
                        <div class="history-item">
                            <div class="history-left">
                                <span class="number"><?php echo $no; ?>.</span>
                                <img src="<?php echo $imgCrop; ?>" alt="Deteksi">
                            </div>
                            <div class="history-right">
                                <p><strong><?php echo $name; ?></strong></p>
                                <p><strong>NIK:</strong> <?php echo $NIK; ?></p>
                                <p><strong>Lokasi Terakhir Terdeteksi:</strong><br><?php echo $loc; ?></p>
                                <p><strong>Waktu:</strong> <?php echo $seenAt; ?></p>
                            </div>
                        </div>
                    <?php
                            $no++;
                        }

                        // Jika tidak ada baris ter-render (misal semua ter-skip)
                        if ($no === 1) {
                            echo '<p>There is no detection history data for NIK <strong>' . htmlspecialchars($nik) . '</strong>.</p>';
                        }
                    } else {
                        if ($nik !== '') {
                            echo '<p>There is no detection history data for NIK <strong>' . htmlspecialchars($nik) . '</strong>.</p>';
                        } else {
                            echo '<p>There is no detection history data yet.</p>';
                        }
                    }
                    ?>
                </div>
            </div>

            <div class="decor_kanan">
                <img src="assets/addition/kanan.png" alt="decor kanan" />
            </div>

            <div class="logo">
                <img src="assets/addition/logo.png" alt="logo" />
            </div>

        </div>
    </div>
</body>
</html>
