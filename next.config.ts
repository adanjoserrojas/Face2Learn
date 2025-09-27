// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Replaces `next export`
  output: 'export',

  // Helpful for Chrome extensions (relative assets)
  trailingSlash: true,

  // If you use next/image
  images: { unoptimized: true },

  // Silence the workspace root warning (optional but useful if you keep parent lockfile)
  // experimental: {
  //   outputFileTracingRoot: __dirname,
  // },
};

module.exports = nextConfig;
