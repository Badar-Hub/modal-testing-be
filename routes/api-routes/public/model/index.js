const express = require("express");
const { model } = require("../../../../controllers/model");
// import { validate as validation } from "~/middlewares";

const router = express.Router();

router.post("/", model);

router.post("/", model);

module.exports = router;
