module.exports = {
  apps: [
    {
      name: "jlyph-frontend",
      script: "npm",
      interpreter: "/mnt/disk2/chenxuan/.nvm/versions/node/v20.18.0/bin/node",
      cwd: "/mnt/disk3/Website/JlyphV2_Frontend",
      args: "run dev",
      env: {
        "NODE_ENV": "development"
      },
    },
    {
      name: "jlyph-backend",
      script: "/mnt/disk3/Website/JlyphV2_Backend/app.py",
      args: "-m flask --app app run --host=0.0.0.0",
      interpreter: "/mnt/disk3/Website/JlyphV2_Backend/jlyph-b/bin/python",
      cwd: "/mnt/disk3/Website/JlyphV2_Backend",

      env: {
        "CUDA_VISIBLE_DEVICES": "1,2",
        "FLASK_ENV": "development",
        "PORT": "9009",
        "PYTHONUNBUFFERED": "1",

        "http_proxy": "http://127.0.0.1:7890",
        "https_proxy": "http://127.0.0.1:7890",
        "HTTP_PROXY": "http://127.0.0.1:7890",
        "HTTPS_PROXY": "http://127.0.0.1:7890",
        
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,.local",
        "NO_PROXY": "localhost,127.0.0.1,0.0.0.0,.local"
      },
    }
  ]
};