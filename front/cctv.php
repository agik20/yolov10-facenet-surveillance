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
    <div class="page-transition">
        <div class="page-content">
            
            <div class="decor_kiri">
                <img src="assets/addition/kiri.png" alt="decor kiri" />
            </div>

            <header data-page="cctv">
                <a href="index.php">Home</a>
                <a href="#" class="orange">CCTV</a>
                <a href="search.php">Search</a>
                <a href="history.php">History</a>
            </header>

            <div class="container">
                <h2>Live CCTV</h2>
                <div class="video-container" id="video_container">
                    <img src="" id="live_video">
                </div>
                <div class="search-box">
                    <input type="text" id="cctvInput" placeholder="Search CCTV Location">
                    <button onclick="searchCCTV()">Search</button>
                </div>
                <div id="cctvResult"></div>
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
