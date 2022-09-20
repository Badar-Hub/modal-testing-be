/* eslint-disable import/named */
/* eslint-disable no-unused-vars */
const express = require("express");
const http = require("http");
const cors = require("cors");
const routes = require("./routes");

const port = process.env.PORT ?? 4000;
const app = express();
app.use(express.json());
app.use(cors());
app.use(routes);

http.createServer(app);
app.listen(port, () => {
  console.log("info", "Info:", `Listening on port: ${port}`);
});

module.exports = app;
