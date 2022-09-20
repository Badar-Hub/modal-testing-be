const express = require("express");
const publicRoutes = require("./api-routes/public");
// import { authorize } from "../middlewares/authorize";

const router = express.Router();

//  API version
// const version = `/${process.env.BASE_URL}/${process.env.VERSION}`;

//  Registering the routes with the router (Public Routes)
publicRoutes.forEach((route) => {
  router.use(route.path, route.route);
});

module.exports = router;
