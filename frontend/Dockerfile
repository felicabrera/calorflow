FROM node:18-alpine

WORKDIR /app

# Copiar package files
COPY package*.json ./

# Instalar dependencias
RUN npm install

# Copiar código
COPY . .

# Exponer puerto
EXPOSE 5173

# Comando para dev server
CMD ["npm", "run", "dev", "--", "--host"]
