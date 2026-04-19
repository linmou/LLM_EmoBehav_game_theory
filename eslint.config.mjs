#!/usr/bin/env node
// Purpose: provide a minimal flat ESLint config so the ESLint MCP can lint repository JavaScript files.

export default [
  {
    ignores: [
      "**/node_modules/**",
      "**/site/**",
      "**/.venv/**",
      "**/__pycache__/**",
    ],
  },
  {
    files: ["**/*.js", "**/*.mjs", "**/*.cjs"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
    },
    rules: {
      "no-undef": "error",
      "no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
        },
      ],
    },
  },
];
