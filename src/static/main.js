import {
  createApp,
  ref,
} from "https://unpkg.com/vue@3/dist/vue.esm-browser.js";

createApp({
  setup() {
    const prediction = ref("");
    return { prediction };
  },
  methods: {
    async initDrawings() {
      /** @type {HTMLCanvasElement} */
      const image = this.$refs.image;
      image.width = 280;
      image.height = 280;
      const ctx = image.getContext("2d");


      let isDrawing = false;
      let pos = { x: 0, y: 0 };

      image.addEventListener("mousedown", (e) => {
        isDrawing = true;
        getPos(e)
      })

      image.addEventListener('mouseup', (e) => {
        isDrawing = false;
      })

      image.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.fillStyle = "#0a0a0a";
        ctx.strokeStyle = "#fafafa";
        ctx.lineCap = 'round';
        ctx.lineWidth = 20;

        ctx.moveTo(pos.x, pos.y);
        getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
      })

      const getPos = (e) => {
        pos.x = e.clientX - image.offsetLeft;
        pos.y = e.clientY - image.offsetTop;
      }
    },

    async predict() {
      /** @type {HTMLCanvasElement} */
      const image = this.$refs.image;

      image.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("image", blob);

        const res = await fetch("/predict", {
          method: "POST",
          body: formData,
        }).then((res) => res.json());

        this.prediction = res.prediction;
      });
    },

    async clear() {
      /** @type {HTMLCanvasElement} */
      const image = this.$refs.image;
      const ctx = image.getContext("2d");
      ctx.clearRect(0, 0, 280, 280);
      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, 280, 280);
      this.prediction = "";
    },
  },
  mounted() {
    this.initDrawings();
  },
}).mount("#app");
