-- phpMyAdmin SQL Dump
-- version 5.2.1deb3
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Aug 21, 2025 at 10:28 AM
-- Server version: 10.11.13-MariaDB-0ubuntu0.24.04.1
-- PHP Version: 8.3.6

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `gotham`
--

-- --------------------------------------------------------

--
-- Table structure for table `cameras`
--

CREATE TABLE `cameras` (
  `camera_id` int(11) NOT NULL,
  `camera_name` varchar(100) NOT NULL,
  `location_name` varchar(200) NOT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `updated_at` datetime DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cameras`
--

INSERT INTO `cameras` (`camera_id`, `camera_name`, `location_name`, `created_at`, `updated_at`) VALUES
(1, 'Cam_001', 'CCTV 01 Kontrakan Utara, Umbulmartani, Kec. Ngemplak, Kabupaten Sleman, Daerah Istimewa Yogyakarta', '2025-08-20 06:57:06', '2025-08-21 09:09:29'),
(2, 'Cam_002', 'Boulevard UII, Kaliurang KM 14.5, Umbulmartani, Kec. Ngemplak, Kabupaten Sleman, Daerah Istimewa Yogyakarta 55584', '2025-08-20 06:57:06', '2025-08-20 06:57:06');

-- --------------------------------------------------------

--
-- Table structure for table `person_last_location`
--

CREATE TABLE `person_last_location` (
  `user_id` int(11) NOT NULL,
  `camera_id` int(11) NOT NULL,
  `last_seen` datetime DEFAULT current_timestamp(),
  `image_path` varchar(255) DEFAULT NULL,
  `full_frame_path` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `person_last_location`
--

INSERT INTO `person_last_location` (`user_id`, `camera_id`, `last_seen`, `image_path`, `full_frame_path`) VALUES
(1, 1, '2025-08-21 16:57:44', '/var/www/html/gotham/assets/output/crop_img/sus_1_cam_1_time_42-57-16.jpg', '/var/www/html/gotham/assets/output/full_img/full_sus_1_cam_1_time_42-57-16.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `person_location_history`
--

CREATE TABLE `person_location_history` (
  `history_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `camera_id` int(11) NOT NULL,
  `seen_at` datetime DEFAULT current_timestamp(),
  `image_path` varchar(255) DEFAULT NULL,
  `full_frame_path` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `person_location_history`
--

INSERT INTO `person_location_history` (`history_id`, `user_id`, `camera_id`, `seen_at`, `image_path`, `full_frame_path`) VALUES
(1, 1, 1, '2025-08-21 16:54:38', '/var/www/html/gotham/assets/output/crop_img/sus_1_cam_1_time_36-54-16.jpg', '/var/www/html/gotham/assets/output/full_img/full_sus_1_cam_1_time_36-54-16.jpg'),
(2, 1, 1, '2025-08-21 16:57:44', '/var/www/html/gotham/assets/output/crop_img/sus_1_cam_1_time_42-57-16.jpg', '/var/www/html/gotham/assets/output/full_img/full_sus_1_cam_1_time_42-57-16.jpg');

-- --------------------------------------------------------

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `user_id` int(11) NOT NULL,
  `NIK` varchar(20) NOT NULL,
  `name` varchar(100) NOT NULL,
  `photo_path` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`user_id`, `NIK`, `name`, `photo_path`) VALUES
(1, '3827194605839217', 'Abyan Nurfajarizqi', '/var/www/html/gotham/assets/id/Abyan.jpg'),
(2, '9164057283945018', 'Hafidz Hidayatullah', '/var/www/html/gotham/assets/id/Apis.jpg'),
(3, '2758930164728593', 'Ardutra Agi Ginting', '/var/www/html/gotham/assets/id/Argi.jpg');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `cameras`
--
ALTER TABLE `cameras`
  ADD PRIMARY KEY (`camera_id`);

--
-- Indexes for table `person_last_location`
--
ALTER TABLE `person_last_location`
  ADD PRIMARY KEY (`user_id`),
  ADD KEY `camera_id` (`camera_id`);

--
-- Indexes for table `person_location_history`
--
ALTER TABLE `person_location_history`
  ADD PRIMARY KEY (`history_id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `camera_id` (`camera_id`);

--
-- Indexes for table `search_requests`
--
ALTER TABLE `search_requests`
  ADD PRIMARY KEY (`search_id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `result_camera_id` (`result_camera_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`user_id`),
  ADD UNIQUE KEY `NIK` (`NIK`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `cameras`
--
ALTER TABLE `cameras`
  MODIFY `camera_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `person_location_history`
--
ALTER TABLE `person_location_history`
  MODIFY `history_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `search_requests`
--
ALTER TABLE `search_requests`
  MODIFY `search_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `user_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `person_last_location`
--
ALTER TABLE `person_last_location`
  ADD CONSTRAINT `person_last_location_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `person_last_location_ibfk_2` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`camera_id`) ON DELETE CASCADE;

--
-- Constraints for table `person_location_history`
--
ALTER TABLE `person_location_history`
  ADD CONSTRAINT `person_location_history_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `person_location_history_ibfk_2` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`camera_id`) ON DELETE CASCADE;

--

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
