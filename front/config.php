<?php
// config.php
$host = "localhost";
$user = "admin";
$pass = "gotham";
$db   = "gotham";

// Koneksi
$mysqli = new mysqli($host, $user, $pass, $db);

// Cek error
if ($mysqli->connect_errno) {
    die("Failed to connect to MySQL: " . $mysqli->connect_error);
}
?>
