const fs = require("fs");
const path = require("path");

const INPUT_DIR = path.join(__dirname, "parsed_texts");
const OUTPUT_DIR = path.join(__dirname, "labeled");
const SHARDED_OUTPUT_DIR = path.join(__dirname, "labeled_100mb");
const MAX_DATASET_BYTES = 100000000;
const TARGET_MIN_SAMPLES = 100000;
const VALID_RATIO = 0.05;

const categoryRules = [
  ["수학", /(수학|해석학|미적분|함수|급수|방정식|정리|소수|분수|숫자|계산|기하|대수)/],
  ["정치", /(정치|정당|대통령|정부|국회|의회|선거|헌법|법률|외교|행정|민주|사회주의)/],
  ["역사", /(역사|세기|왕조|전쟁|혁명|제국|독립|근대|고대|중세|현대|사건)/],
  ["경제", /(경제|금융|투자|주식|시장|기업|산업|무역|화폐|가격|자본|생산)/],
  ["의학/바이오", /(의학|질병|바이러스|세균|백신|치료|증상|병원|생명|바이오|유전자)/],
  ["과학", /(과학|물리|화학|생물|천문|우주|원자|분자|에너지|진화|실험)/],
  ["기술", /(기술|컴퓨터|인공지능|AI|소프트웨어|하드웨어|인터넷|프로그래밍|알고리즘|반도체|전자)/i],
  ["사회", /(사회|문화|교육|가족|노동|인권|범죄|지역|도시|인구|언론)/],
  ["예술/문화", /(음악|영화|드라마|문학|소설|만화|애니|게임|예술|미술|공연|작품)/],
  ["스포츠", /(스포츠|축구|야구|농구|배구|선수|경기|리그|올림픽|월드컵)/],
];

const noisePatterns = [
  /최근 변경/,
  /최근 토론/,
  /특수 기능/,
  /최근 수정 시각/,
  /편집 권한/,
  /편집 요청/,
  /편집 보호/,
  /ACL 탭/,
  /확인하세요/,
  /토론/,
  /역사/,
  /v-if/,
  /펼치기 · 접기/,
  /닫기/,
  /분류/,
  /나무위키/,
  /map\.naver\.com/i,
  /internetsignal\.shop/i,
  /https?:\/\//i,
  /www\./i,
  /Operado por/i,
  /Hecho con/i,
  /Asunción/i,
  /República del Paraguay/i,
  /umanle/i,
  /the seed/i,
  /CC BY/i,
  /정성을 담은 수제요리/,
  /특별한 날엔/,
  /인터넷가입/,
  /신규가입/,
  /가입할인/,
  /결합특가/,
  /지원금/,
  /전국최대/,
  /실시간스포츠/,
  /당일설치/,
  /케이블/,
  /Su zona horaria/i,
  /GMT/,
  /기여하신 문서/,
  /저작권/,
  /reCAPTCHA/i,
  /hCaptcha/i,
  /Privacy Policy/i,
  /Terms of Service/i,
];

const metadataPatterns = [
  /^(약칭|영어 명칭|한국어 명칭|창당일|이념|스펙트럼|의장|국내 조직|국제 조직|웹사이트)/,
  /^(국회의원|라틴아메리카 의회|재적|여당|야당)/,
  /^\d+\s*석\s*\/\s*\d+\s*석$/,
];

function normalizeWhitespace(text) {
  return text.replace(/\s+/g, " ").trim();
}

function escapeLine(text) {
  return text.replace(/[<>]/g, "").replace(/\r?\n/g, " ").trim();
}

function getTitle(fileName, text) {
  return path.basename(fileName, ".txt").replace(/_/g, "/").trim();
}

function detectCategory(title, text) {
  const target = `${title}\n${text.slice(0, 3000)}`;
  for (const [category, pattern] of categoryRules) {
    if (pattern.test(target)) {
      return category;
    }
  }
  return "일반";
}

function cleanLine(line) {
  let cleaned = line
    .replace(/\[[^\S\r\n]*\]/g, " ")
    .replace(/\[|\]/g, " ")
    .replace(/\s*\|\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  cleaned = cleaned.replace(/^[*-]\s*/, "").trim();
  cleaned = cleaned.replace(/^\d+(\.\d+)*\s*\.\s*/, "").trim();
  cleaned = cleaned.replace(/\s*\[편집\]\s*/g, " ").trim();
  return cleaned;
}

function isUsefulLine(line) {
  if (line.length < 24 || line.length > 700) {
    return false;
  }
  if (noisePatterns.some((pattern) => pattern.test(line))) {
    return false;
  }
  if (metadataPatterns.some((pattern) => pattern.test(line))) {
    return false;
  }
  const koreanCount = (line.match(/[가-힣]/g) || []).length;
  if (koreanCount < 5) {
    return false;
  }
  const koreanOrAlpha = (line.match(/[가-힣A-Za-z0-9]/g) || []).length;
  return koreanOrAlpha / line.length >= 0.45;
}

function splitSentences(line) {
  const sentences = line
    .split(/(?<=[.!?。！？]|다\.|요\.|임\.|음\.|됨\.|함\.)\s+/)
    .map(normalizeWhitespace)
    .filter((sentence) => sentence.length >= 18 && sentence.length <= 350);

  if (sentences.length) {
    return sentences;
  }
  return [line];
}

function chunkParagraphs(paragraphs) {
  const chunks = [];

  for (const paragraph of paragraphs) {
    for (const sentence of splitSentences(paragraph)) {
      chunks.push(sentence);
    }
  }

  for (let i = 0; i < paragraphs.length; i += 1) {
    const current = paragraphs[i];
    const next = paragraphs[i + 1];
    if (next && current.length + next.length <= 650) {
      chunks.push(`${current} ${next}`);
    }
  }

  return chunks;
}

function makeSamples(title, category, chunks) {
  const samples = [];

  for (const chunk of chunks) {
    const content = escapeLine(chunk);
    if (!content) {
      continue;
    }

    samples.push(`<s> 다음 ${category} 내용을 설명하라: ${content} </s>`);
    samples.push(`<s> 질문: ${title}에 대해 어떤 내용인가요? 답변: ${content} </s>`);

    if (content.length >= 80) {
      samples.push(`<s> 질문: ${title} 문서에서 중요한 내용은 무엇인가요? 답변: ${content} </s>`);
    }
  }

  return samples;
}

function dedupePreserveOrder(items) {
  const seen = new Set();
  const result = [];

  for (const item of items) {
    if (seen.has(item)) {
      continue;
    }
    seen.add(item);
    result.push(item);
  }

  return result;
}

function clearDatasetShards(outputDir, prefix) {
  if (!fs.existsSync(outputDir)) {
    return;
  }

  for (const file of fs.readdirSync(outputDir)) {
    if (new RegExp(`^${prefix}_\\d{3}\\.txt$`).test(file)) {
      fs.unlinkSync(path.join(outputDir, file));
    }
  }
}

function writeShardedDataset(outputDir, prefix, samples) {
  fs.mkdirSync(outputDir, { recursive: true });
  clearDatasetShards(outputDir, prefix);

  let index = 1;
  let currentBytes = 0;
  let currentLines = [];
  const written = [];

  function flush() {
    if (!currentLines.length) {
      return;
    }

    const fileName = `${prefix}_${String(index).padStart(3, "0")}.txt`;
    const filePath = path.join(outputDir, fileName);
    fs.writeFileSync(filePath, `${currentLines.join("\n")}\n`, "utf8");
    written.push(filePath);
    index += 1;
    currentBytes = 0;
    currentLines = [];
  }

  for (const sample of samples) {
    const lineBytes = Buffer.byteLength(`${sample}\n`, "utf8");
    if (currentLines.length && currentBytes + lineBytes > MAX_DATASET_BYTES) {
      flush();
    }

    currentLines.push(sample);
    currentBytes += lineBytes;
  }

  flush();
  return written;
}
function buildDataset() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(SHARDED_OUTPUT_DIR, { recursive: true });

  const files = fs.readdirSync(INPUT_DIR).filter((file) => file.endsWith(".txt"));
  const samples = [];

  for (const file of files) {
    const filePath = path.join(INPUT_DIR, file);
    const text = fs.readFileSync(filePath, "utf8");
    const title = getTitle(file, text);
    const cleanedLines = text.split(/\r?\n+/).map(cleanLine);
    const overviewIndex = cleanedLines.findIndex((line) => /^개요(\s|$)/.test(line));
    const contentLines = overviewIndex >= 0 ? cleanedLines.slice(overviewIndex + 1) : cleanedLines;
    const paragraphs = contentLines.filter(isUsefulLine);
    const category = detectCategory(title, paragraphs.join("\n"));

    const chunks = dedupePreserveOrder(chunkParagraphs(paragraphs));
    samples.push(...makeSamples(title, category, chunks));
  }

  const uniqueSamples = dedupePreserveOrder(samples);
  if (uniqueSamples.length < TARGET_MIN_SAMPLES) {
    console.warn(`Warning: only ${uniqueSamples.length} unique samples were generated.`);
  }

  const validCount = Math.max(1, Math.floor(uniqueSamples.length * VALID_RATIO));
  const valid = uniqueSamples.filter((_, index) => index % Math.floor(1 / VALID_RATIO) === 0).slice(0, validCount);
  const validSet = new Set(valid);
  const train = uniqueSamples.filter((sample) => !validSet.has(sample));

  const trainShards = writeShardedDataset(SHARDED_OUTPUT_DIR, "train", train);
  const validShards = writeShardedDataset(SHARDED_OUTPUT_DIR, "valid", valid);

  console.log(`documents=${files.length}`);
  console.log(`samples=${uniqueSamples.length}`);
  console.log(`train=${train.length}`);
  console.log(`valid=${valid.length}`);
  console.log(`train_shards=${trainShards.length}`);
  console.log(`valid_shards=${validShards.length}`);
  console.log(`output=${SHARDED_OUTPUT_DIR}`);
}

buildDataset();
