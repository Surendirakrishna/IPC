const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const VERSION = '20260101';
const SECTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
const BASE_URL = `https://www.wipo.int/ipc/itos4ipc/ITSupport_and_download_area/${VERSION}/IPC_scheme_title_list`;

/**
 * Download a URL and return the content as a string
 */
function downloadUrl(url) {
  return new Promise((resolve, reject) => {
    const handler = (response) => {
      // Handle redirects
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        const redirectUrl = response.headers.location.startsWith('http')
          ? response.headers.location
          : new URL(response.headers.location, url).href;
        console.log(`  Redirected to: ${redirectUrl}`);
        const protocol = redirectUrl.startsWith('https') ? https : http;
        protocol.get(redirectUrl, handler).on('error', reject);
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode} for ${url}`));
        return;
      }
      let data = '';
      response.on('data', chunk => data += chunk);
      response.on('end', () => resolve(data));
      response.on('error', reject);
    };
    https.get(url, handler).on('error', reject);
  });
}

/**
 * Parse IPC symbol to extract its hierarchical components.
 * IPC Hierarchy:
 *   Section:  A (1 letter)
 *   Class:    A01 (1 letter + 2 digits)
 *   Subclass: A01B (1 letter + 2 digits + 1 letter)
 *   Group:    A01B0001000000 (main group when last 6 digits are 000000)
 *   Subgroup: A01B0001020000 (subgroup when last 6 digits != 000000)
 */
function classifySymbol(symbol) {
  symbol = symbol.trim();

  // Section: single letter (A, B, C, ...)
  if (/^[A-H]$/.test(symbol)) {
    return { type: 'section', symbol };
  }

  // Class: letter + 2 digits (A01, B23, ...)
  if (/^[A-H]\d{2}$/.test(symbol)) {
    return { type: 'class', symbol, section: symbol[0] };
  }

  // Subclass: letter + 2 digits + letter (A01B, C12N, ...)
  if (/^[A-H]\d{2}[A-Z]$/.test(symbol)) {
    return { type: 'subclass', symbol, class: symbol.substring(0, 3) };
  }

  // Group/Subgroup: full IPC symbol (A01B0001000000)
  // Format: 4-char subclass + 4-digit maingroup + 6-digit subgroup
  if (/^[A-H]\d{2}[A-Z]\d{10}$/.test(symbol)) {
    const subclass = symbol.substring(0, 4);
    const mainGroupNum = symbol.substring(4, 8);
    const subGroupNum = symbol.substring(8, 14);

    if (subGroupNum === '000000') {
      return {
        type: 'maingroup',
        symbol,
        subclass,
        mainGroupNum,
      };
    } else {
      return {
        type: 'subgroup',
        symbol,
        subclass,
        mainGroupNum,
        subGroupNum,
        parentMainGroup: `${subclass}${mainGroupNum}000000`,
      };
    }
  }

  // Unknown format
  return { type: 'unknown', symbol };
}

/**
 * Determine the depth of a subgroup based on the number pattern.
 * This groups subgroups hierarchically based on common prefixes in their subgroup number.
 */
function getSubgroupDepth(subGroupNum) {
  // Count trailing zeros to determine depth
  // e.g., 020000 → depth 1, 020100 → depth 2, 020101 → depth 3
  const str = subGroupNum;
  if (str.endsWith('00000')) return 1;
  if (str.endsWith('0000')) return 2;
  if (str.endsWith('000')) return 3;
  if (str.endsWith('00')) return 4;
  if (str.endsWith('0')) return 5;
  return 6;
}

/**
 * Parse all downloaded data and build the tree
 */
function buildTree(allEntries) {
  const root = {
    symbol: 'IPC',
    title: 'International Patent Classification (CPC Scheme)',
    version: VERSION,
    children: [],
  };

  // Maps for quick lookup
  const sectionMap = {};
  const classMap = {};
  const subclassMap = {};
  const mainGroupMap = {};

  // First pass: create all nodes
  for (const entry of allEntries) {
    const { symbol, title } = entry;
    const info = classifySymbol(symbol);

    const node = {
      symbol,
      title,
      type: info.type,
      children: [],
    };

    switch (info.type) {
      case 'section':
        sectionMap[symbol] = node;
        root.children.push(node);
        break;

      case 'class':
        classMap[symbol] = node;
        if (sectionMap[info.section]) {
          sectionMap[info.section].children.push(node);
        }
        break;

      case 'subclass':
        subclassMap[symbol] = node;
        if (classMap[info.class]) {
          classMap[info.class].children.push(node);
        }
        break;

      case 'maingroup':
        mainGroupMap[symbol] = node;
        if (subclassMap[info.subclass]) {
          subclassMap[info.subclass].children.push(node);
        }
        break;

      case 'subgroup': {
        // Subgroups go under their main group
        // We build a flat list under the main group first
        const parentKey = info.parentMainGroup;
        if (mainGroupMap[parentKey]) {
          mainGroupMap[parentKey].children.push(node);
        } else {
          // Some subgroups might reference main groups we haven't seen yet
          // Try to find the subclass and add there
          if (subclassMap[info.subclass]) {
            subclassMap[info.subclass].children.push(node);
          }
        }
        break;
      }

      default:
        // Skip unknown entries
        console.warn(`  Unknown symbol format: "${symbol}" → "${title}"`);
        break;
    }
  }

  return root;
}

/**
 * Clean up the tree: remove empty children arrays and type fields for leaves
 */
function cleanTree(node) {
  if (node.children && node.children.length === 0) {
    delete node.children;
  } else if (node.children) {
    for (const child of node.children) {
      cleanTree(child);
    }
  }
  return node;
}

/**
 * Count nodes in tree
 */
function countNodes(node) {
  let count = 1;
  if (node.children) {
    for (const child of node.children) {
      count += countNodes(child);
    }
  }
  return count;
}

async function main() {
  console.log('=== WIPO IPC/CPC Scheme Extractor ===\n');
  console.log(`Version: ${VERSION}`);
  console.log(`Sections: ${SECTIONS.join(', ')}\n`);

  const allEntries = [];

  // Download each section
  for (const section of SECTIONS) {
    const fileName = `EN_ipc_section_${section}_title_list_${VERSION}.txt`;
    const url = `${BASE_URL}/${fileName}`;
    console.log(`Downloading Section ${section}...`);
    console.log(`  URL: ${url}`);

    try {
      const data = await downloadUrl(url);
      const lines = data.split('\n').filter(line => line.trim());
      let count = 0;

      for (const line of lines) {
        const tabIndex = line.indexOf('\t');
        if (tabIndex === -1) continue;

        const symbol = line.substring(0, tabIndex).trim();
        const title = line.substring(tabIndex + 1).trim();

        if (symbol && title) {
          allEntries.push({ symbol, title });
          count++;
        }
      }

      console.log(`  ✓ Parsed ${count} entries from Section ${section}\n`);
    } catch (err) {
      console.error(`  ✗ Error downloading Section ${section}: ${err.message}\n`);
    }
  }

  console.log(`Total entries parsed: ${allEntries.length}\n`);

  // Build tree
  console.log('Building hierarchical tree...');
  const tree = buildTree(allEntries);
  cleanTree(tree);

  const totalNodes = countNodes(tree);
  console.log(`Total nodes in tree: ${totalNodes}`);

  // Summary
  console.log('\nTree Summary:');
  for (const section of tree.children || []) {
    const classCount = (section.children || []).length;
    console.log(`  Section ${section.symbol}: "${section.title}" — ${classCount} classes`);
  }

  // Write output
  const outputPath = path.join(__dirname, 'ipc_schema_tree.json');
  fs.writeFileSync(outputPath, JSON.stringify(tree, null, 2), 'utf-8');
  console.log(`\n✓ Schema written to: ${outputPath}`);

  // Also write a compact version
  const compactPath = path.join(__dirname, 'ipc_schema_tree_compact.json');
  fs.writeFileSync(compactPath, JSON.stringify(tree), 'utf-8');
  console.log(`✓ Compact schema written to: ${compactPath}`);

  const stats = fs.statSync(outputPath);
  const compactStats = fs.statSync(compactPath);
  console.log(`\nFile sizes:`);
  console.log(`  Pretty-printed: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  Compact:        ${(compactStats.size / 1024 / 1024).toFixed(2)} MB`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
