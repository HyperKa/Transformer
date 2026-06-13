# Стейдж 1: Сборка
FROM maven:3.9.6-eclipse-temurin-22-alpine AS build
WORKDIR /app
COPY pom.xml .
COPY src ./src
# Сборка проекта
RUN mvn clean package -DskipTests

# Стейдж 2: Запуск
FROM eclipse-temurin:22-jre-alpine
WORKDIR /app

# Копируем JAR файл.
# Используем маску, но указываем конкретное целевое имя файла БЕЗ слэша.
COPY --from build /app/target/*.jar ./app.jar

# Создаем пустые файлы, чтобы Docker не создал папки вместо них при монтировании volumes
RUN touch checkpoint.txt final_training_dataset.txt

# Запуск с ограничением памяти
ENTRYPOINT ["java", "-Xmx4G", "-jar", "app.jar"]