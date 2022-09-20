const modelRoutes = require("./model");

const publicRoutes = [
  {
    path: "/test",
    route: modelRoutes,
  },
];

module.exports = publicRoutes;
