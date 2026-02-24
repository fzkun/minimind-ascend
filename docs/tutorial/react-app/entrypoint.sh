#!/bin/sh
# 启动后端（FastAPI :8999）
python3 /app/scripts/serve_train_manager.py &

# 委托给 nginx 官方入口（处理 envsubst 模板替换后启动 nginx）
exec /docker-entrypoint.sh nginx -g 'daemon off;'
