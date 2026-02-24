# MiniMind Web 前端 + 后端一体化镜像
# 构建方式：由 run_all_npu.sh stage_web 自动执行
#   1. npm run build → 生成 dist/
#   2. 复制 serve_train_manager.py 到构建上下文
#   3. docker build
FROM m.daocloud.io/docker.io/library/nginx:alpine

# Python + 后端依赖
RUN apk add --no-cache python3 py3-pip \
    && pip3 install --no-cache-dir --break-system-packages \
       fastapi uvicorn pydantic

# 前端静态文件
COPY dist/ /usr/share/nginx/html/

# Nginx 配置模板（envsubst 替换 ${WEB_PORT}）
COPY nginx.conf /etc/nginx/templates/default.conf.template

# 后端
COPY serve_train_manager.py /app/scripts/serve_train_manager.py

# 入口脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV WEB_PORT=80
ENTRYPOINT ["/entrypoint.sh"]
