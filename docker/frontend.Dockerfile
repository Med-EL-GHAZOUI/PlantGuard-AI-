FROM nginxinc/nginx-unprivileged:1.27-alpine

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/proxy.conf /etc/nginx/plantguard_proxy.conf
COPY frontend /usr/share/nginx/html
# même origine que l'API (proxy nginx) : adresse relative, pas de CORS à ouvrir
# (COPY et non RUN : l'image « unprivileged » n'a pas le droit d'écrire dans ce dossier)
COPY docker/config.js /usr/share/nginx/html/config.js

EXPOSE 8080
