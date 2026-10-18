// Render docs/re_study/report/content.json as a Word document.
//
//   NODE_PATH=<dir with node_modules/docx> node docs/re_study/report/build_docx.js
//
// content.json is written by build_content.py from the stored result files.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, ImageRun, LevelFormat, BorderStyle, Footer, PageNumber,
} = require("docx");

const HERE = __dirname;
const blocks = JSON.parse(fs.readFileSync(path.join(HERE, "content.json"), "utf8"));
const OUT = path.join(HERE, "..", "re_study_results.docx");

const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134;          // A4, 2 cm margins
const CONTENT_W = PAGE_W - 2 * MARGIN;
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };

function scaled(widths, n) {
  const w = widths && widths.length === n ? widths : Array(n).fill(1);
  const total = w.reduce((a, b) => a + b, 0);
  const out = w.map((x) => Math.floor((x * CONTENT_W) / total));
  out[out.length - 1] += CONTENT_W - out.reduce((a, b) => a + b, 0);
  return out;
}

function cell(text, width, header) {
  return new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: "E8EEF7", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: header, size: 16 })] })],
  });
}

function table(t) {
  const n = t.head.length;
  const widths = scaled(t.widths, n);
  const rows = [new TableRow({ tableHeader: true, children: t.head.map((h, i) => cell(h, widths[i], true)) })];
  for (const r of t.rows) rows.push(new TableRow({ children: r.map((c, i) => cell(c, widths[i], false)) }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows });
}

const children = [];
for (const b of blocks) {
  if (b.title) children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(b.title)] }));
  else if (b.h1) children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(b.h1)] }));
  else if (b.h2) children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(b.h2)] }));
  else if (b.p) children.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun(b.p)] }));
  else if (b.note) children.push(new Paragraph({ spacing: { before: 60, after: 200 },
    children: [new TextRun({ text: b.note, italics: true, size: 17, color: "52514E" })] }));
  else if (b.bullets) for (const item of b.bullets)
    children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun(item)] }));
  else if (b.table) { children.push(table(b.table)); if (!blocks[blocks.indexOf(b) + 1] || !blocks[blocks.indexOf(b) + 1].note)
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] })); }
  else if (b.img) {
    // fit the page: no wider than the text, no taller than 85% of it (room for the caption)
    const maxW = Math.floor((CONTENT_W / 1440) * 96);
    const maxH = Math.floor(((PAGE_H - 2 * MARGIN) / 1440) * 96 * 0.85);
    const k = Math.min(1, maxW / b.w, maxH / b.h);
    const w = Math.round(b.w * k), h = Math.round(b.h * k);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.img),
        transformation: { width: w, height: h }, altText: { title: "figure", description: b.caption, name: path.basename(b.img) } })] }));
    children.push(new Paragraph({ spacing: { after: 240 },
      children: [new TextRun({ text: b.caption, italics: true, size: 18 })] }));
  }
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 40, bold: true, font: "Calibri" },
        paragraph: { spacing: { after: 240 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Calibri", color: "1F3864" },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Calibri", color: "1F3864" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H },
      margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 16 })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote " + OUT); });
