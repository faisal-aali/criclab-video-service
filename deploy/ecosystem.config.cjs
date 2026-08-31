module.exports = {
  apps: [
    {
      name: "criclab-video-api",
      cwd: "/var/www/criclab-video-service",
      script: "/var/www/criclab-video-service/deploy/start-api.sh",
      interpreter: "bash",
      instances: 1,
      exec_mode: "fork",
      autorestart: true,
      max_memory_restart: "512M",
    },
    {
      name: "criclab-video-worker",
      cwd: "/var/www/criclab-video-service",
      script: "/var/www/criclab-video-service/deploy/start-worker.sh",
      interpreter: "bash",
      instances: 2,
      exec_mode: "fork",
      autorestart: true,
      max_memory_restart: "2G",
      kill_timeout: 30000,
    },
  ],
}
