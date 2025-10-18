# 🐳 Docker — Краткая справка

## Основные команды

### Управление образами

```


# Список образов

docker images

# Удалить образ

docker rmi <image_id>

# Сохранить образ в архив

docker save ct-pathology-api:latest | gzip > ct-pathology-api.tar.gz

# Загрузить образ из архива

docker load < ct-pathology-api.tar.gz

```

### Управление контейнерами

```


# Список запущенных контейнеров

docker ps

# Список всех контейнеров (включая остановленные)

docker ps -a

# Запуск контейнера

docker start <container_name>

# Остановка контейнера

docker stop <container_name>

# Удаление контейнера

docker rm <container_name>

# Логи контейнера

docker logs <container_name>

```

### Подключение к контейнеру

```


# Запуск bash внутри контейнера

docker exec -it ct-pathology-api bash

# Выполнение команды в контейнере

docker exec ct-pathology-api ls -la /app/models

```

### Очистка системы

```


# Удалить неиспользуемые контейнеры

docker container prune

# Удалить неиспользуемые образы

docker image prune

# Полная очистка (осторожно!)

docker system prune -a

```

## Работа с volumes

```


# Монтирование локальной директории

docker run -v /host/path:/container/path <image>

# Создание именованного volume

docker volume create ct-data
docker run -v ct-data:/app/data <image>

# Список volumes

docker volume ls

# Удалить volume

docker volume rm ct-data
