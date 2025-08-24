const chokidar = require("chokidar");
const mysql = require("mysql2/promise");
const path = require("path");

// ================== CONFIG ==================
const config = {
  WATCH_DIR_CROP: "/var/www/html/gotham/assets/output/crop_img",
  WATCH_DIR_FULL: "/var/www/html/gotham/assets/output/full_img",
  DB_CONFIG: {
    host: "localhost",
    user: "admin",
    password: "gotham",
    database: "gotham",
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0,
  },
  ALLOWED_EXTENSIONS: [".jpg", ".jpeg", ".png"],
  TIMEZONE: "Asia/Jakarta", // WIB
  PAIR_TIMEOUT_MS: 2000, // waktu tunggu pairing crop<->full
};

// ================== UTIL ==================
const toMysqlDatetimeTz = (date, timeZone) => {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
    hour12: false,
  })
    .formatToParts(date)
    .reduce((acc, p) => ((acc[p.type] = p.value), acc), {});
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`;
};

const getNowWib = () => toMysqlDatetimeTz(new Date(), config.TIMEZONE);

// ================== DATABASE SERVICE ==================
class DatabaseService {
  constructor() {
    this.pool = mysql.createPool(config.DB_CONFIG);
  }

  async withTransaction(workFn) {
    const conn = await this.pool.getConnection();
    try {
      await conn.query("SET time_zone = '+07:00'"); // WIB
      await conn.beginTransaction();
      const result = await workFn(conn);
      await conn.commit();
      return result;
    } catch (e) {
      await conn.rollback();
      throw e;
    } finally {
      conn.release();
    }
  }

  async insertHistory(conn, userId, cameraId, cropPath, fullPath, seenAt) {
    await conn.query(
      `INSERT INTO person_location_history 
         (user_id, camera_id, image_path, full_frame_path, seen_at)
       VALUES (?, ?, ?, ?, ?)`,
      [userId, cameraId, cropPath, fullPath, seenAt]
    );
  }

  async upsertLastLocation(conn, userId, cameraId, cropPath, fullPath, seenAt) {
    await conn.query(
      `INSERT INTO person_last_location 
         (user_id, camera_id, image_path, full_frame_path, last_seen)
       VALUES (?, ?, ?, ?, ?)
       ON DUPLICATE KEY UPDATE
         camera_id = VALUES(camera_id),
         image_path = VALUES(image_path),
         full_frame_path = VALUES(full_frame_path),
         last_seen = VALUES(last_seen)`,
      [userId, cameraId, cropPath, fullPath, seenAt]
    );
  }

  async close() {
    await this.pool.end();
  }
}

// ================== FILE SERVICE ==================
class FileService {
  static parseCrop(filePath) {
    const base = path.basename(filePath);
    // Format: sus_{userId}_cam_{camId}_time_{SS-MM-HH}.jpg
    const match = base.match(/^sus_(\d+)_cam_(\d+)_time_(\d{2})-(\d{2})-(\d{2})\.(jpg|jpeg|png)$/i);
    if (!match) return null;

    return {
      user_id: parseInt(match[1], 10),
      camera_id: parseInt(match[2], 10),
      crop_path: filePath,
      timestamp: {
        second: parseInt(match[3], 10),
        minute: parseInt(match[4], 10),
        hour: parseInt(match[5], 10),
      }
    };
  }

  static parseFull(filePath) {
    const base = path.basename(filePath);
    // Format: full_sus_{userId}_cam_{camId}_time_{SS-MM-HH}.jpg
    const match = base.match(/^full_sus_(\d+)_cam_(\d+)_time_(\d{2})-(\d{2})-(\d{2})\.(jpg|jpeg|png)$/i);
    if (!match) return null;

    return {
      user_id: parseInt(match[1], 10),
      camera_id: parseInt(match[2], 10),
      full_path: filePath,
      timestamp: {
        second: parseInt(match[3], 10),
        minute: parseInt(match[4], 10),
        hour: parseInt(match[5], 10),
      }
    };
  }
}


// ================== WATCHER PIPELINE ==================
class FileWatcher {
  constructor() {
    this.db = new DatabaseService();
    this.pending = new Map(); // cache pairing sementara

    this.watcherCrop = chokidar.watch(config.WATCH_DIR_CROP, {
      ignoreInitial: true,
      persistent: true,
      awaitWriteFinish: { stabilityThreshold: 2000, pollInterval: 100 },
    });

    this.watcherFull = chokidar.watch(config.WATCH_DIR_FULL, {
      ignoreInitial: true,
      persistent: true,
      awaitWriteFinish: { stabilityThreshold: 2000, pollInterval: 100 },
    });
  }

  makeKey(userId, camId) {
    return `${userId}_${camId}`;
  }

  async handlePair(userId, cameraId, cropPath, fullPath) {
    const seenAt = getNowWib();
    await this.db.withTransaction(async (conn) => {
      await this.db.insertHistory(
        conn, userId, cameraId, cropPath, fullPath, seenAt
      );
      await this.db.upsertLastLocation(
        conn, userId, cameraId, cropPath, fullPath, seenAt
      );
    });
    console.log(`DB updated → user ${userId}, cam ${cameraId}`);
  }

  processCrop(info) {
    const key = this.makeKey(info.user_id, info.camera_id);
    let entry = this.pending.get(key) || {};
    entry.crop = info.crop_path;

    if (entry.full) {
      clearTimeout(entry.timer);
      this.pending.delete(key);
      this.handlePair(info.user_id, info.camera_id, entry.crop, entry.full);
    } else {
      entry.timer = setTimeout(() => {
        this.pending.delete(key);
        this.handlePair(info.user_id, info.camera_id, entry.crop, null);
      }, config.PAIR_TIMEOUT_MS);
      this.pending.set(key, entry);
    }
  }

  processFull(info) {
    const key = this.makeKey(info.user_id, info.camera_id);
    let entry = this.pending.get(key) || {};
    entry.full = info.full_path;

    if (entry.crop) {
      clearTimeout(entry.timer);
      this.pending.delete(key);
      this.handlePair(info.user_id, info.camera_id, entry.crop, entry.full);
    } else {
      entry.timer = setTimeout(() => {
        this.pending.delete(key);
        console.log(`Full frame ${info.full_path} tidak ada crop → dilewati`);
      }, config.PAIR_TIMEOUT_MS);
      this.pending.set(key, entry);
    }
  }

  async initialize() {
    this.setupWatcher();
    this.setupShutdown();
    console.log(`Memantau folder: ${config.WATCH_DIR_CROP} & ${config.WATCH_DIR_FULL}`);
  }

  setupWatcher() {
    this.watcherCrop.on("add", (filePath) => {
      const ext = path.extname(filePath).toLowerCase();
      if (!config.ALLOWED_EXTENSIONS.includes(ext)) return;
      const info = FileService.parseCrop(filePath);
      if (!info) return;
      console.log(`Crop file masuk: ${filePath}`);
      this.processCrop(info);
    });

    this.watcherFull.on("add", (filePath) => {
      const ext = path.extname(filePath).toLowerCase();
      if (!config.ALLOWED_EXTENSIONS.includes(ext)) return;
      const info = FileService.parseFull(filePath);
      if (!info) return;
      console.log(`Full file masuk: ${filePath}`);
      this.processFull(info);
    });
  }

  setupShutdown() {
    const cleanup = async () => {
      console.log("Menutup watcher...");
      try {
        await this.watcherCrop.close();
        await this.watcherFull.close();
        await this.db.close();
        console.log("Watcher & DB ditutup dengan bersih");
        process.exit(0);
      } catch (err) {
        console.error("Kesalahan saat shutdown:", err);
        process.exit(1);
      }
    };
    process.on("SIGINT", cleanup);
    process.on("SIGTERM", cleanup);
  }
}

// ================== MAIN ==================
(async () => {
  try {
    const watcher = new FileWatcher();
    await watcher.initialize();
  } catch (err) {
    console.error("Gagal memulai watcher:", err);
    process.exit(1);
  }
})();
