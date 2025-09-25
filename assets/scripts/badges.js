const fs = require('fs');

const repo_raw = process.env.REPO || process.env.GITHUB_REPOSITORY || '';
const repo = repo_raw.trim();
const IS_PRIVATE = String(process.env.IS_PRIVATE) === 'true';

const publicBadge = `
<div align="center">
  <a href="https://github.com/${repo.split('/')[0]}" target="_blank" rel="noopener noreferrer">
    <img alt="Made by Soph" src="https://img.shields.io/badge/Made%20by-Soph-ff69b4?style=for-the-badge" />
  </a>
  <a href="https://github.com/${repo}/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
    <img alt="MIT License" src="https://img.shields.io/github/license/${repo}?style=for-the-badge" />
  </a>
  <a href="https://github.com/${repo}" target="_blank" rel="noopener noreferrer">
    <img alt="Last commit" src="https://img.shields.io/github/last-commit/${repo}?style=for-the-badge" />
  </a>
  <a href="https://github.com/${repo}" target="_blank" rel="noopener noreferrer">
    <img alt="Repo size" src="https://img.shields.io/github/repo-size/${repo}?style=for-the-badge" />
  </a>
</div>`;

const privateBadge = `
<div align="center">
  <a href="https://github.com/${repo.split('/')[0]}" target="_blank" rel="noopener noreferrer">
    <img alt="Made by Soph" src="https://img.shields.io/badge/Made%20by-Soph-ff69b4?style=for-the-badge" />
  </a>
  <a href="https://github.com/${repo}/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge" />
  </a>
  <a href="https://github.com/${repo}/commits" target="_blank" rel="noopener noreferrer">
    <img alt="Last commit" src="https://img.shields.io/badge/last%20commit-see%20history-informational?style=for-the-badge" />
  </a>
  <img alt="Repo size" src="https://img.shields.io/badge/repo%20size-private-lightgrey?style=for-the-badge" />
</div>`;

const html = IS_PRIVATE ? privateBadge : publicBadge;

const path = 'README.md';
let readme = fs.readFileSync(path, 'utf8');

const start = '<!-- Badge Start -->';
const end = '<!-- Badge End -->';
const block = `${start}\n${html}\n${end}`;

if (readme.includes(start) && readme.includes(end)) {
  readme = readme.replace(new RegExp(`${start}[\\s\\S]*?${end}`, 'm'), block);
} else {
  readme = `${block}\n\n${readme}`;
}

const before = fs.readFileSync(path, 'utf8');
if (before !== readme) {
  fs.writeFileSync(path, readme);
  console.log('README updated with', IS_PRIVATE ? 'private' : 'public', 'badges.');
} else {
  console.log('No README change needed.');
}