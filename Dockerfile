# Сборка проекта (используем оптимизацию под linux)
FROM maven:3.9-eclipse-temurin-22-alpine AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline -Djavacpp.platform=linux-x86_64
COPY src ./src
RUN mvn clean package -DskipTests -Djavacpp.platform=linux-x86_64

# Запуск
FROM eclipse-temurin:22-jre-alpine
WORKDIR /app
# Копируем созданный shade-плагином "толстый" jar
COPY --from=build /app/target/*.jar ./app.jar

# Запускаем напрямую через -jar
ENTRYPOINT ["java", "-jar", "app.jar"]