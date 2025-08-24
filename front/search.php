<?php
// Koneksi ke database
require_once 'config.php';

// Default: tidak ada user yang dicari
$last = null;
$history = null;

// Jika ada pencarian NIK lewat GET
if (isset($_GET['nik']) && $_GET['nik'] !== '') {
    $nik = $mysqli->real_escape_string($_GET['nik']);

    // Cari user_id dari NIK
    $sql_user = "SELECT user_id FROM users WHERE NIK = '$nik' LIMIT 1";
    $res_user = $mysqli->query($sql_user);
    if ($res_user && $res_user->num_rows > 0) {
        $user = $res_user->fetch_assoc();
        $user_id = $user['user_id'];

        // Ambil data lokasi terakhir
        $sql_last = "
            SELECT pll.*, u.name, u.NIK, c.location_name, u.photo_path
            FROM person_last_location pll
            JOIN users u ON pll.user_id = u.user_id
            JOIN cameras c ON pll.camera_id = c.camera_id
            WHERE pll.user_id = $user_id
        ";
        $last = $mysqli->query($sql_last)->fetch_assoc();
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Gotham</title>
    <link rel="stylesheet" href="style.css">
    <script src="script.js" defer></script>
</head>
<body>
    <div class="page-transition">
        <div class="page-content">
            
            <div class="decor_kiri">
                <img src="assets/addition/kiri.png" alt="decor kiri" />
            </div>

            <header data-page="search">
                <a href="index.php">Home</a>
                <a href="cctv.php">CCTV</a>
                <a href="#" class="active">Search</a>
                <a href="history.php">History</a>
            </header>

            <div class="container">
                <div class="search-box">
                    <form method="get" action="">
                        <input type="text" name="nik" placeholder="Search by ID (NIK)" 
                               value="<?php echo isset($_GET['nik']) ? htmlspecialchars($_GET['nik']) : ''; ?>">
                        <button type="submit">Search</button>
                    </form>
                </div>

                <div class="result-box">
                    <?php if ($last): ?>
                        <div class="profile">
                            <img src="<?php echo str_replace('/var/www/html/', '/', $last['photo_path']); ?>" 
                                 alt="Foto ID" width="200">
                            <h2><?php echo htmlspecialchars($last['name']); ?></h2>
                            <p><?php echo htmlspecialchars($last['NIK']); ?></p>
                        </div>

                        <div class="divider"></div>

                        <div class="detection">
                            <div class="detection-images">
                                <img src="<?php echo str_replace('/var/www/html/', '/', $last['full_frame_path']); ?>" 
                                     alt="Full Frame" width="400">
                                <img src="<?php echo str_replace('/var/www/html/', '/', $last['image_path']); ?>" 
                                     alt="Deteksi Terakhir" width="150">
                            </div>
                            <p><strong>Lokasi Terakhir Terdeteksi:</strong><br>
                               <?php echo htmlspecialchars($last['location_name']); ?>
                            </p>
                        </div>
                    <?php elseif (isset($_GET['nik'])): ?>
                        <p>Data not found for NIK
                           <strong><?php echo htmlspecialchars($_GET['nik']); ?></strong>.
                        </p>
                    <?php else: ?>
                        <p>Please enter NIK to search for data.</p>
                    <?php endif; ?>
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

