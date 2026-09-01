module.exports = {
  apps: [
    {
      name: "criclab-video-worker",
      cwd: "/var/www/criclab-video-service",
      script: "/var/www/criclab-video-service/deploy/start-worker.sh",
      interpreter: "bash",
      instances: 1,
      exec_mode: "fork",
      autorestart: true,
      // TODO: Need to add a fallback mechansim for inprogress jobs, Need to increase memory to 4gb
      // max_memory_restart: "2G",
      kill_timeout: 30000,
    },
  ],
}
