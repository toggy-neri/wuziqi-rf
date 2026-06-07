const BOARD_SIZE = 15;
const BLACK = 1;
const WHITE = -1;
const MODEL_URL = './alphaTao-v1.0.onnx';

const TEXT = {
  loading: '\u6a21\u578b\u52a0\u8f7d\u4e2d',
  playerTurn: '\u73a9\u5bb6\u56de\u5408',
  aiThinking: 'AI \u601d\u8003\u4e2d',
  modelFailed: '\u6a21\u578b\u52a0\u8f7d\u5931\u8d25',
  playerWin: '\u73a9\u5bb6\u80dc',
  aiWin: 'AI \u80dc',
  draw: '\u5e73\u5c40',
  player: '\u73a9\u5bb6',
  ai: 'AI',
  black: '\u9ed1',
  white: '\u767d',
  valueGood: '\u6b63\u503c\u5229\u4e8e AI',
  valueBad: '\u8d1f\u503c\u5229\u4e8e\u73a9\u5bb6',
  valueEven: '\u5c40\u9762\u63a5\u8fd1\u5747\u52bf',
};

const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const valueEl = document.getElementById('valueScore');
const valueHintEl = document.getElementById('valueHint');
const thinkEl = document.getElementById('thinkTime');
const levelEl = document.getElementById('level');
const playerColorEl = document.getElementById('playerColor');
const blackRoleEl = document.getElementById('blackRole');
const whiteRoleEl = document.getElementById('whiteRole');
const resetBtn = document.getElementById('resetBtn');
const undoBtn = document.getElementById('undoBtn');

let session = null;
let board = createBoard();
let moveHistory = [];
let humanPlayer = BLACK;
let aiPlayer = WHITE;
let currentPlayer = BLACK;
let gameOver = false;
let thinking = false;
let lastValue = null;
let lastThinkMs = null;

function createBoard() {
  return Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(0));
}

function setStatus(text, tone = 'neutral') {
  statusEl.textContent = text;
  statusEl.dataset.tone = tone;
}

function syncPlayersFromControl() {
  humanPlayer = playerColorEl.value === 'white' ? WHITE : BLACK;
  aiPlayer = -humanPlayer;
}

function updateRoles() {
  blackRoleEl.textContent = humanPlayer === BLACK ? TEXT.player : TEXT.ai;
  whiteRoleEl.textContent = humanPlayer === WHITE ? TEXT.player : TEXT.ai;
}

function actionOf(row, col) {
  return row * BOARD_SIZE + col;
}

function pointOf(action) {
  return [Math.floor(action / BOARD_SIZE), action % BOARD_SIZE];
}

function inBounds(row, col) {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

function isEmpty(action) {
  const [row, col] = pointOf(action);
  return board[row][col] === 0;
}

function place(action, player) {
  const [row, col] = pointOf(action);
  board[row][col] = player;
  moveHistory.push({ action, player });
  currentPlayer = -player;
}

function resetGame() {
  syncPlayersFromControl();
  board = createBoard();
  moveHistory = [];
  currentPlayer = BLACK;
  gameOver = false;
  thinking = false;
  lastValue = null;
  lastThinkMs = null;
  updateRoles();
  updatePanel();
  drawBoard();
  if (!session) {
    setStatus(TEXT.loading);
    return;
  }
  if (currentPlayer === aiPlayer) {
    void aiMove();
  } else {
    setStatus(TEXT.playerTurn);
  }
}

function undoPair() {
  if (thinking || moveHistory.length === 0) return;
  do {
    const move = moveHistory.pop();
    const [row, col] = pointOf(move.action);
    board[row][col] = 0;
    currentPlayer = move.player;
  } while (moveHistory.length && currentPlayer !== humanPlayer);
  gameOver = false;
  lastValue = null;
  lastThinkMs = null;
  updatePanel();
  drawBoard();
  setStatus(currentPlayer === humanPlayer ? TEXT.playerTurn : TEXT.aiThinking);
  if (session && currentPlayer === aiPlayer) void aiMove();
}

function checkWinner(lastAction) {
  if (lastAction == null) return null;
  const [row, col] = pointOf(lastAction);
  const player = board[row][col];
  if (!player) return null;
  const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
  for (const [dr, dc] of dirs) {
    let count = 1;
    for (const sign of [1, -1]) {
      let r = row + dr * sign;
      let c = col + dc * sign;
      while (inBounds(r, c) && board[r][c] === player) {
        count += 1;
        r += dr * sign;
        c += dc * sign;
      }
    }
    if (count >= 5) return player;
  }
  return board.flat().every(Boolean) ? 0 : null;
}

function winningMoves(player) {
  const wins = [];
  for (let action = 0; action < BOARD_SIZE * BOARD_SIZE; action += 1) {
    if (!isEmpty(action)) continue;
    const [row, col] = pointOf(action);
    board[row][col] = player;
    if (checkWinner(action) === player) wins.push(action);
    board[row][col] = 0;
  }
  return wins;
}

function validMask(radius = 2) {
  const mask = new Float32Array(BOARD_SIZE * BOARD_SIZE);
  const occupied = [];
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      if (board[r][c] !== 0) occupied.push([r, c]);
    }
  }

  if (occupied.length === 0) {
    const center = Math.floor(BOARD_SIZE / 2);
    for (let r = center - radius; r <= center + radius; r += 1) {
      for (let c = center - radius; c <= center + radius; c += 1) {
        if (inBounds(r, c)) mask[actionOf(r, c)] = 1;
      }
    }
    return mask;
  }

  for (const [row, col] of occupied) {
    for (let r = row - radius; r <= row + radius; r += 1) {
      for (let c = col - radius; c <= col + radius; c += 1) {
        if (inBounds(r, c) && board[r][c] === 0) mask[actionOf(r, c)] = 1;
      }
    }
  }
  return mask;
}

function buildInput(player) {
  const data = new Float32Array(1 * 8 * BOARD_SIZE * BOARD_SIZE);
  const plane = (channel, row, col, value) => {
    data[channel * BOARD_SIZE * BOARD_SIZE + row * BOARD_SIZE + col] = value;
  };

  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      if (board[r][c] === BLACK) plane(0, r, c, 1);
      if (board[r][c] === WHITE) plane(1, r, c, 1);
      plane(7, r, c, player);
    }
  }

  for (let i = 0; i < 5; i += 1) {
    const move = moveHistory[moveHistory.length - 1 - i];
    if (!move) continue;
    const [row, col] = pointOf(move.action);
    plane(2 + i, row, col, 1);
  }

  return new ort.Tensor('float32', data, [1, 8, BOARD_SIZE, BOARD_SIZE]);
}

function softmaxMasked(logits, mask) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    if (mask[i] > 0 && logits[i] > max) max = logits[i];
  }
  const probs = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    if (mask[i] <= 0) continue;
    const v = Math.exp(logits[i] - max);
    probs[i] = v;
    sum += v;
  }
  if (sum > 0) {
    for (let i = 0; i < probs.length; i += 1) probs[i] /= sum;
  }
  return probs;
}

function chooseFromPolicy(probs) {
  const ranked = Array.from(probs, (prob, action) => ({ action, prob }))
    .filter(item => item.prob > 0)
    .sort((a, b) => b.prob - a.prob);
  if (ranked.length === 0) return firstValidAction();

  const level = levelEl.value;
  if (level === 'strong') return ranked[0].action;

  const topK = level === 'fast' ? 8 : 4;
  const temperature = level === 'fast' ? 1.35 : 0.85;
  const pool = ranked.slice(0, Math.min(topK, ranked.length));
  const weights = pool.map(item => Math.pow(item.prob, 1 / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  let roll = Math.random() * total;
  for (let i = 0; i < pool.length; i += 1) {
    roll -= weights[i];
    if (roll <= 0) return pool[i].action;
  }
  return pool[0].action;
}

function firstValidAction() {
  const mask = validMask();
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i] > 0) return i;
  }
  return board.flat().findIndex(cell => cell === 0);
}

async function aiMove() {
  if (!session || gameOver || currentPlayer !== aiPlayer) return;
  thinking = true;
  setStatus(TEXT.aiThinking);
  updatePanel();
  await new Promise(resolve => setTimeout(resolve, 20));

  const started = performance.now();
  const win = winningMoves(aiPlayer);
  const block = winningMoves(humanPlayer);
  let action;

  if (win.length) {
    action = win[0];
    lastValue = 1;
  } else if (block.length) {
    action = block[0];
    lastValue = null;
  } else {
    const feeds = { state: buildInput(aiPlayer) };
    const output = await session.run(feeds);
    const policy = output.policy || output[session.outputNames[0]];
    const value = output.value || output[session.outputNames[1]];
    const probs = softmaxMasked(policy.data, validMask());
    action = chooseFromPolicy(probs);
    lastValue = Number(value.data[0]);
  }

  lastThinkMs = performance.now() - started;
  place(action, aiPlayer);
  const winner = checkWinner(action);
  if (winner !== null) finishGame(winner);
  else setStatus(TEXT.playerTurn);
  thinking = false;
  updatePanel();
  drawBoard();
}

function finishGame(winner) {
  gameOver = true;
  if (winner === humanPlayer) setStatus(TEXT.playerWin, 'win');
  else if (winner === aiPlayer) setStatus(TEXT.aiWin, 'loss');
  else setStatus(TEXT.draw);
}

function updateValueHint() {
  if (lastValue == null) {
    valueHintEl.textContent = TEXT.valueGood;
  } else if (lastValue > 0.08) {
    valueHintEl.textContent = TEXT.valueGood;
  } else if (lastValue < -0.08) {
    valueHintEl.textContent = TEXT.valueBad;
  } else {
    valueHintEl.textContent = TEXT.valueEven;
  }
}

function updatePanel() {
  valueEl.textContent = lastValue == null ? '--' : lastValue.toFixed(3);
  updateValueHint();
  thinkEl.textContent = lastThinkMs == null ? '--' : `${Math.round(lastThinkMs)}ms`;
  undoBtn.disabled = thinking || moveHistory.length === 0;
}

function drawBoard() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.width * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const size = rect.width;
  const pad = size * 0.06;
  const gap = (size - pad * 2) / (BOARD_SIZE - 1);

  ctx.clearRect(0, 0, size, size);
  const gradient = ctx.createLinearGradient(0, 0, size, size);
  gradient.addColorStop(0, '#e3bd68');
  gradient.addColorStop(1, '#bf873c');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  ctx.strokeStyle = 'rgba(45, 34, 21, 0.82)';
  ctx.lineWidth = 1;
  for (let i = 0; i < BOARD_SIZE; i += 1) {
    const p = pad + i * gap;
    ctx.beginPath();
    ctx.moveTo(pad, p);
    ctx.lineTo(size - pad, p);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p, pad);
    ctx.lineTo(p, size - pad);
    ctx.stroke();
  }

  for (const [r, c] of [[3, 3], [3, 11], [7, 7], [11, 3], [11, 11]]) {
    ctx.beginPath();
    ctx.arc(pad + c * gap, pad + r * gap, Math.max(3, gap * 0.08), 0, Math.PI * 2);
    ctx.fillStyle = '#4a3520';
    ctx.fill();
  }

  const last = moveHistory[moveHistory.length - 1];
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      const player = board[r][c];
      if (!player) continue;
      const x = pad + c * gap;
      const y = pad + r * gap;
      const radius = gap * 0.42;
      const piece = ctx.createRadialGradient(x - radius * 0.35, y - radius * 0.35, radius * 0.1, x, y, radius);
      if (player === BLACK) {
        piece.addColorStop(0, '#555');
        piece.addColorStop(1, '#050505');
      } else {
        piece.addColorStop(0, '#fff');
        piece.addColorStop(1, '#d8d3c7');
      }
      ctx.fillStyle = piece;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      if (last && last.action === actionOf(r, c)) {
        ctx.strokeStyle = player === BLACK ? '#f4d35e' : '#0f766e';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    }
  }
}

function clickToAction(event) {
  const rect = canvas.getBoundingClientRect();
  const size = rect.width;
  const pad = size * 0.06;
  const gap = (size - pad * 2) / (BOARD_SIZE - 1);
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const col = Math.round((x - pad) / gap);
  const row = Math.round((y - pad) / gap);
  const px = pad + col * gap;
  const py = pad + row * gap;
  if (!inBounds(row, col)) return null;
  if (Math.hypot(x - px, y - py) > gap * 0.45) return null;
  return actionOf(row, col);
}

canvas.addEventListener('click', async event => {
  if (thinking || gameOver || currentPlayer !== humanPlayer || !session) return;
  const action = clickToAction(event);
  if (action == null || !isEmpty(action)) return;
  place(action, humanPlayer);
  const winner = checkWinner(action);
  drawBoard();
  updatePanel();
  if (winner !== null) {
    finishGame(winner);
    updatePanel();
    return;
  }
  await aiMove();
});

resetBtn.addEventListener('click', resetGame);
undoBtn.addEventListener('click', undoPair);
playerColorEl.addEventListener('change', resetGame);
window.addEventListener('resize', drawBoard);

async function boot() {
  if (window.lucide) window.lucide.createIcons();
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    if (currentPlayer === aiPlayer) await aiMove();
    else setStatus(TEXT.playerTurn);
  } catch (error) {
    console.error(error);
    setStatus(TEXT.modelFailed, 'loss');
  }
  updatePanel();
  drawBoard();
}

resetGame();
boot();
