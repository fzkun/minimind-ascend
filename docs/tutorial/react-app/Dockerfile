# 前端静态服务 + 反向代理
# 构建方式：先在宿主机 npm run build，再 docker build
#   cd docs/tutorial/react-app && npm run build && docker build -t minimind-web .
FROM m.daocloud.io/docker.io/library/nginx:alpine
COPY dist/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/templates/default.conf.template
ENV WEB_PORT=80
