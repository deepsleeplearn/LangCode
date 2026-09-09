import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { diffLines } from 'diff';
import {
  ArrowDown,
  Bot,
  ChevronDown,
  ChevronRight,
  Check,
  CircleAlert,
  Code2,
  Copy,
  Folder,
  AudioLines,
  Monitor,
  Moon,
  MoreHorizontal,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Send,
  Settings,
  SlidersHorizontal,
  Square,
  Sun,
  Terminal,
  X,
} from 'lucide-react';
import './styles.css';

const API_TOKEN = document.querySelector('meta[name="langcode-token"]')?.content || '';
const API_HEADERS = { 'X-LangCode-Token': API_TOKEN };
const DISPLAY_DELTA_FLUSH_MS = 64;
const TOAST_TTL_MS = 6000;
const TOAST_DEDUPE_MS = 5000;
const TOAST_LIMIT = 5;
// Two-stage barge-in: the backend VAD `speech start` only ducks, a confirmed intent stops.
const TTS_DUCK_VOLUME = 0.2;
const BARGE_IN_CONFIRM_WINDOW_MS = 1500;
// Upper bound on holding the auto-end timer open for a `speech start` whose `end` never came.
const VOICE_SPEECH_HOLD_MAX_MS = 8000;
const TTS_PLAYED_AUDIO_KEY_LIMIT = 512;
const VOICE_METRICS_TURN_LIMIT = 8;

// Components rendered outside <App> (markdown code blocks) still need to report failures.
let toastSink = null;
let activeTranslations = null;

function emitToast(payload) {
  toastSink?.(payload);
}

const THEME_STORAGE_KEY = 'langcode-theme';
const THEME_MODES = ['system', 'light', 'dark'];

// Safari private mode and "block all cookies" make every localStorage access throw. An
// unguarded read inside a useState initializer takes the whole page down, so every access
// goes through these two helpers.
function safeStorageGet(key, fallback = '') {
  try {
    const stored = localStorage.getItem(key);
    return stored === null ? fallback : stored;
  } catch {
    return fallback;
  }
}

function safeStorageSet(key, value) {
  try {
    if (value === null || value === undefined) localStorage.removeItem(key);
    else localStorage.setItem(key, String(value));
  } catch {
    // Storage may be unavailable (private mode); the app still works for this tab.
  }
}

function readStoredThemeMode() {
  const stored = safeStorageGet(THEME_STORAGE_KEY, 'system');
  return THEME_MODES.includes(stored) ? stored : 'system';
}

function writeStoredThemeMode(mode) {
  safeStorageSet(THEME_STORAGE_KEY, mode);
}

function applyThemeAttribute(mode) {
  const root = document.documentElement;
  if (mode === 'light' || mode === 'dark') root.dataset.theme = mode;
  else delete root.dataset.theme;
}

function systemPrefersDark() {
  return Boolean(window.matchMedia?.('(prefers-color-scheme: dark)').matches);
}

function isEffectiveDark(mode) {
  return mode === 'dark' || (mode !== 'light' && systemPrefersDark());
}

applyThemeAttribute(readStoredThemeMode());

let effectiveDarkTheme = isEffectiveDark(readStoredThemeMode());
const themeListeners = new Set();

function subscribeThemeChange(listener) {
  themeListeners.add(listener);
  return () => themeListeners.delete(listener);
}

function publishThemeChange(nextDark) {
  if (nextDark === effectiveDarkTheme) return;
  effectiveDarkTheme = nextDark;
  mermaidModulePromise = null;
  mermaidThemeDark = null;
  for (const listener of themeListeners) listener(nextDark);
}

const darkThemeMediaQuery = window.matchMedia?.('(prefers-color-scheme: dark)');
darkThemeMediaQuery?.addEventListener?.('change', () => {
  if (readStoredThemeMode() === 'system') publishThemeChange(systemPrefersDark());
});

function mermaidConfig(dark) {
  return {
    startOnLoad: false,
    securityLevel: 'strict',
    theme: dark ? 'dark' : 'base',
    themeVariables: dark ? undefined : {
      fontFamily: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      primaryColor: '#f7f7f7',
      primaryBorderColor: '#d1d1d1',
      primaryTextColor: '#0d0d0d',
      lineColor: '#8f8f8f',
      secondaryColor: '#eef8f5',
      tertiaryColor: '#ffffff',
    },
  };
}

let mermaidModulePromise = null;
let mermaidThemeDark = null;

function loadMermaid(dark = effectiveDarkTheme) {
  if (!mermaidModulePromise || mermaidThemeDark !== dark) {
    mermaidThemeDark = dark;
    mermaidModulePromise = import('mermaid').then((module) => {
      const instance = module.default;
      instance.initialize(mermaidConfig(dark));
      return instance;
    });
  }
  return mermaidModulePromise;
}

const MATH_MARKUP_PATTERN = /\$|\\\(|\\\[/;
const CODE_FENCE_PATTERN = /```/;
const BASE_REMARK_PLUGINS = [remarkGfm];
const EMPTY_REHYPE_PLUGINS = [];

let mathPluginsCache = null;
let mathPluginsPromise = null;
let highlightPluginCache = null;
let highlightPluginPromise = null;

// A rejected import must not be cached: otherwise one flaky chunk fetch disables math or
// highlighting for the rest of the page's life. Resetting the slot lets the next message retry.
function loadMathPlugins() {
  if (!mathPluginsPromise) {
    mathPluginsPromise = import('./markdown-math.jsx')
      .then((module) => {
        mathPluginsCache = { remarkMath: module.remarkMath, rehypeKatex: module.rehypeKatex };
        return mathPluginsCache;
      })
      .catch((err) => {
        mathPluginsPromise = null;
        throw err;
      });
  }
  return mathPluginsPromise;
}

function loadHighlightPlugin() {
  if (!highlightPluginPromise) {
    highlightPluginPromise = import('./markdown-highlight.jsx')
      .then((module) => {
        highlightPluginCache = [module.rehypeHighlight, { ignoreMissing: true, detect: false }];
        return highlightPluginCache;
      })
      .catch((err) => {
        highlightPluginPromise = null;
        throw err;
      });
  }
  return highlightPluginPromise;
}

const MODEL_OPTIONS = [
  {
    value: 'openai:glm-5:aimp-glm',
    provider: 'openai',
    model: 'glm-5',
    gateway: 'aimp-glm',
    label: 'AIMP GLM-5',
    supportsThinking: true,
  },
  { value: 'openai:gpt-4o:aimp', provider: 'openai', model: 'gpt-4o', gateway: 'aimp', label: 'AIMP GPT-4o' },
  {
    value: 'openai:deepseek-v4-pro:aimp-deepseek-v4-pro',
    provider: 'openai',
    model: 'deepseek-v4-pro',
    gateway: 'aimp-deepseek-v4-pro',
    label: 'AIMP DeepSeek-V4 Pro',
    supportsThinking: true,
  },
];

const DEFAULT_TTS_VOICE_OPTION = {
  id: 'default',
  name: '默认音色',
  style: 'macOS 系统音色',
  builtIn: true,
  previewReady: true,
  previewUrl: '/api/tts/voices/default/preview',
  previewText: '欢迎使用LangCode，你的最后一个智能体。',
};

const TTS_VOICE_ALIASES = {
  汪菊: 'wangju',
  雪芬: 'xuefen',
};

function ttsVoiceIdentity(voice) {
  const id = String(voice?.id || '').trim();
  const name = String(voice?.name || '').trim();
  return (TTS_VOICE_ALIASES[id] || TTS_VOICE_ALIASES[name] || id || name).toLowerCase();
}

function preferTtsVoice(candidate, current) {
  if (!current) return true;
  if (candidate.id === DEFAULT_TTS_VOICE_OPTION.id) return true;
  const candidateScore =
    (candidate.provider === 'mlx-cosyvoice3' ? 8 : 0) +
    (candidate.previewReady ? 4 : 0) +
    (candidate.profileReady ? 2 : 0) +
    (candidate.builtIn ? 1 : 0);
  const currentScore =
    (current.provider === 'mlx-cosyvoice3' ? 8 : 0) +
    (current.previewReady ? 4 : 0) +
    (current.profileReady ? 2 : 0) +
    (current.builtIn ? 1 : 0);
  return candidateScore > currentScore;
}

function normalizeTtsVoiceOptions(voices = []) {
  const normalized = Array.isArray(voices) ? voices.filter((voice) => voice?.id) : [];
  const output = normalized.map((voice) =>
    voice.id === DEFAULT_TTS_VOICE_OPTION.id ? { ...DEFAULT_TTS_VOICE_OPTION, ...voice, previewUrl: voice.previewUrl || DEFAULT_TTS_VOICE_OPTION.previewUrl, previewReady: true } : voice,
  );
  const withDefault = output.some((voice) => voice.id === DEFAULT_TTS_VOICE_OPTION.id) ? output : [DEFAULT_TTS_VOICE_OPTION, ...output];
  const deduped = new Map();
  for (const voice of withDefault) {
    const identity = ttsVoiceIdentity(voice);
    if (!identity) continue;
    const current = deduped.get(identity);
    if (preferTtsVoice(voice, current)) {
      deduped.set(identity, voice);
    }
  }
  return Array.from(deduped.values());
}

const SLASH_COMMANDS = [
  {
    command: '/compact',
    labels: { en: 'Compact context', zh: '压缩上下文' },
    descriptions: {
      en: 'Summarize older messages and archive the full pre-compact history.',
      zh: '汇总较早消息，并归档压缩前的完整历史。',
    },
  },
  {
    command: '/memory',
    labels: { en: 'Show memory', zh: '查看记忆' },
    descriptions: {
      en: 'Show loaded project memory and instructions.',
      zh: '显示已加载的项目记忆和指令。',
    },
  },
  {
    command: '/agents',
    labels: { en: 'Show agents', zh: '查看子 Agent' },
    descriptions: {
      en: 'List built-in read-only sub-agents.',
      zh: '列出内置只读子 Agent。',
    },
  },
  {
    command: '/skills',
    labels: { en: 'Show skills', zh: '查看技能' },
    descriptions: {
      en: 'Show project skills loaded from .langcode/skills.',
      zh: '显示从 .langcode/skills 加载的项目技能。',
    },
  },
];

const TRANSLATIONS = {
  en: {
    welcome: 'How can I help with this codebase?',
    codeAgent: 'Code agent',
    newSession: 'New session',
    chooseWorkspaceFirst: 'Choose a workspace before starting a new chat.',
    sessionWorkspace: 'Session workspace',
    chats: 'Chats',
    renameSession: 'Rename session',
    renameSessionPrompt: 'Session name',
    clearSession: 'Clear session',
    deleteSession: 'Delete session',
    hideSidebar: 'Hide sidebar',
    showSidebar: 'Show sidebar',
    provider: 'Provider',
    loading: 'loading',
    modelPending: 'model pending',
    keyReady: 'Model key configured',
    toolMode: 'Tool mode available',
    workspaceReady: 'Workspace ready',
    chatHeader: 'Ask, inspect, edit, and run code from one place.',
    refreshStatus: 'Refresh status',
    working: 'Working...',
    voiceTtsPreparing: 'Preparing voice response...',
    processingApproval: 'Processing approval...',
    processingReadFile: 'Reading file...',
    processingWriteFile: 'Writing file...',
    processingEditFile: 'Editing file...',
    processingSearch: 'Searching...',
    processingShell: 'Running command...',
    processingTool: 'Running {tool}...',
    progressTitle: 'Agent progress',
    planTitle: 'Plan',
    activityTitle: 'Activity',
    thinkingTitle: 'Thinking',
    thinkingDoneTitle: 'Thought process',
    thinkingPlaceholder: 'Thinking through the task...',
    todoStatus: {
      pending: 'pending',
      in_progress: 'in progress',
      completed: 'done',
      blocked: 'blocked',
      cancelled: 'cancelled',
    },
    progressCompleted: '{count} instructions completed',
    progressNext: 'Summarizing results and deciding next step.',
    progressWaitingApproval: 'Waiting for approval',
    progressActions: {
      read_file: 'Reading file',
      ls: 'Listing directory',
      glob: 'Finding files',
      write_file: 'Writing file',
      edit_file: 'Editing file',
      search: 'Searching',
      web_search: 'Searching web',
      web_fetch: 'Reading webpage',
      shell: 'Running command',
      sandbox_shell: 'Running sandbox command',
      task_create: 'Creating task',
      task_update: 'Updating task',
      task_list: 'Reading tasks',
      task_get: 'Reading task',
      task_cancel: 'Cancelling task',
      delegate_agent: 'Running sub-agent',
      diagram: 'Drawing diagram',
    },
    allowTool: 'Allow {tool}?',
    reviewToolInput: 'Review tool input before it runs.',
    toolInput: 'Tool input',
    approvalEditLabel: 'Approval edit or feedback',
    accept: 'Accept',
    acceptRemember: 'Accept & remember',
    edit: 'Edit',
    feedback: 'Feedback',
    reject: 'Reject',
    forceEnd: 'Force end',
    pendingPlaceholder: 'Resolve the approval above before sending another message...',
    askPlaceholder: 'Ask LangCode to inspect, edit, run, or explain code...',
    sendMessage: 'Send message',
    stopGeneration: 'Stop generation',
    jumpToLatest: 'Jump to latest',
    slashCommands: 'Commands',
    slashCommandHint: 'Use arrow keys and Enter to select.',
    currentWorkspace: 'Current workspace',
    workspaceInputLabel: 'Workspace directory',
    nativePickerPrompt: 'Choose workspace directory',
    close: 'Close',
    retry: 'Retry',
    theme: 'Theme',
    themeSystem: 'Follow system',
    themeLight: 'Light',
    themeDark: 'Dark',
    heartbeatWaiting: 'Model responding… {seconds}s',
    usageLine: 'In {input} · Out {output}',
    contextCompacted: 'Context compacted',
    toolResultPreview: 'Tool output',
    toolResultTruncated: '(truncated)',
    sessionCompleted: 'Finished while you were away',
    errorAuth: 'Check the API key configuration',
    errorRateLimit: 'Rate limited, try again later',
    errorModelTimeout: 'The model did not respond; you can retry',
    errorContextOverflow: 'The session is too long — run /compact or start a new session',
    errorNetwork: 'Network error',
    settings: 'Settings',
    model: 'Model',
    displayLanguage: 'Display language',
    thinkingMode: 'Thinking mode',
    thinkingModeHint: 'Only applies to AIMP GLM-5 and AIMP DeepSeek-V4 Pro.',
    voiceInputModel: 'Voice input model',
    voiceModelQwen: 'Qwen3-ASR-0.6B',
    voiceFeature: 'Voice input',
    voiceInterrupt: 'Interrupt with voice',
    voiceStarting: 'Starting voice input...',
    voiceListening: 'Listening...',
    voiceTranscribing: 'Transcribing...',
    voiceFinalizing: 'Finalizing voice input...',
    voiceReady: 'Voice input ready',
    voiceUnavailable: 'Voice input is unavailable: {error}',
    voicePermissionDenied: 'Microphone permission was denied or unavailable.',
    stopVoice: 'Stop voice input',
    ttsEnabled: 'Voice playback',
    ttsVoice: 'Assistant voice',
    previewVoice: 'Preview voice',
    voicePreviewFailed: 'Voice preview failed: {error}',
    bargeInListening: 'Listening for interruption...',
    bargeInDetected: 'Interruption detected. Stopping playback...',
    english: 'English',
    chinese: 'Chinese',
    saveSettings: 'Save',
    invalidApprovalJson: 'Edited tool input is invalid JSON: {error}',
    copyFailed: 'Copy failed: {error}',
    staleTokenReloadFailed:
      'The server rejected this page (unauthorized). Reload the page; if it keeps failing, restart the server.',
    diffTooLarge: 'Content too large, diff omitted (before {old} chars, after {new} chars).',
    requestFailed: 'Request failed',
    streamUnavailable: 'Streaming is not available in this browser.',
    streamFailed: 'Stream failed: {error}',
    rejectedFromUi: 'Rejected from web UI',
    reviseTryAgain: 'Revise and try again.',
    workspaceChanged: 'Workspace switched. Started a fresh session for the new directory.',
  },
  zh: {
    welcome: '我可以怎样帮你处理这个代码库？',
    codeAgent: '代码 Agent',
    newSession: '新会话',
    chooseWorkspaceFirst: '请先为新会话选择工作目录。',
    sessionWorkspace: '会话工作目录',
    chats: '会话',
    renameSession: '重命名',
    renameSessionPrompt: '会话名称',
    clearSession: '清空会话',
    deleteSession: '删除会话',
    hideSidebar: '关闭侧边栏',
    showSidebar: '打开侧边栏',
    provider: '模型服务',
    loading: '加载中',
    modelPending: '模型待加载',
    keyReady: '模型密钥已配置',
    toolMode: '可使用工具模式',
    workspaceReady: '工作目录已就绪',
    chatHeader: 'Your Last Agent.',
    refreshStatus: '刷新状态',
    working: '处理中...',
    voiceTtsPreparing: '正在合成语音...',
    processingApproval: '正在处理审批',
    processingReadFile: '正在读取文件',
    processingWriteFile: '正在写入文件',
    processingEditFile: '正在编辑文件',
    processingSearch: '正在搜索',
    processingShell: '正在执行命令',
    processingTool: '正在执行 {tool}',
    progressTitle: '执行进度',
    planTitle: '任务计划',
    activityTitle: '执行动态',
    thinkingTitle: '正在思考',
    thinkingDoneTitle: '思考过程',
    thinkingPlaceholder: '正在分析问题、约束和下一步...',
    todoStatus: {
      pending: '待办',
      in_progress: '进行中',
      completed: '已完成',
      blocked: '阻塞',
      cancelled: '已取消',
    },
    progressCompleted: '已执行 {count} 条指令',
    progressNext: '正在整理结果并决定下一步。',
    progressWaitingApproval: '等待审批',
    progressActions: {
      read_file: '正在读取文件',
      ls: '正在列目录',
      glob: '正在查找文件',
      write_file: '正在写入文件',
      edit_file: '正在编辑文件',
      search: '正在搜索',
      web_search: '正在搜索网页',
      web_fetch: '正在读取网页',
      shell: '正在执行命令',
      sandbox_shell: '正在沙箱执行命令',
      task_create: '正在创建任务',
      task_update: '正在更新任务',
      task_list: '正在读取任务',
      task_get: '正在读取任务',
      task_cancel: '正在取消任务',
      delegate_agent: '正在调用子 Agent',
      diagram: '正在绘制图示',
    },
    allowTool: '允许执行 {tool}？',
    reviewToolInput: '执行前请确认工具输入。',
    toolInput: '工具输入',
    approvalEditLabel: '审批修改或反馈',
    accept: '允许',
    acceptRemember: '允许并记住',
    edit: '修改后执行',
    feedback: '反馈',
    reject: '拒绝',
    forceEnd: '强制结束',
    pendingPlaceholder: '请先处理上方审批项，再发送新消息...',
    askPlaceholder: '有问题、有需求，尽管提',
    sendMessage: '发送消息',
    stopGeneration: '停止生成',
    jumpToLatest: '回到最新',
    slashCommands: '命令',
    slashCommandHint: '可用方向键和回车选择。',
    currentWorkspace: '当前工作目录',
    workspaceInputLabel: '工作目录',
    nativePickerPrompt: '选择工作目录',
    close: '关闭',
    retry: '重试',
    theme: '主题',
    themeSystem: '跟随系统',
    themeLight: '浅色',
    themeDark: '深色',
    heartbeatWaiting: '模型响应中… {seconds} 秒',
    usageLine: '输入 {input} · 输出 {output}',
    contextCompacted: '上下文已压缩',
    toolResultPreview: '工具输出',
    toolResultTruncated: '（已截断）',
    sessionCompleted: '离开期间已完成',
    errorAuth: '检查 API key 配置',
    errorRateLimit: '稍后重试',
    errorModelTimeout: '模型无响应，可重试',
    errorContextOverflow: '会话过长，请 /compact 或新建会话',
    errorNetwork: '网络错误',
    settings: '设置',
    model: '模型',
    displayLanguage: '显示语言',
    thinkingMode: '开启 Thinking',
    thinkingModeHint: '仅对 AIMP GLM-5 和 AIMP DeepSeek-V4 Pro 生效。',
    voiceInputModel: '语音输入模型',
    voiceModelQwen: 'Qwen3-ASR-0.6B',
    voiceFeature: '语音功能',
    voiceInterrupt: '语音打断',
    voiceStarting: '正在启动语音输入...',
    voiceListening: '正在聆听...',
    voiceTranscribing: '正在实时转写...',
    voiceFinalizing: '正在固定转写内容...',
    voiceReady: '语音输入已就绪',
    voiceUnavailable: '语音输入不可用：{error}',
    voicePermissionDenied: '麦克风权限被拒绝或不可用。',
    stopVoice: '停止语音输入',
    ttsEnabled: '语音播报',
    ttsVoice: '回复音色',
    previewVoice: '试听音色',
    voicePreviewFailed: '试听失败：{error}',
    bargeInListening: '正在监听是否打断...',
    bargeInDetected: '检测到打断意图，正在停止播报...',
    english: '英文',
    chinese: '中文',
    saveSettings: '保存',
    invalidApprovalJson: '修改后的工具输入不是有效 JSON：{error}',
    copyFailed: '复制失败：{error}',
    staleTokenReloadFailed: '服务端拒绝了当前页面（未授权）。请刷新页面；如果仍然失败，请重启服务。',
    diffTooLarge: '内容过大，已省略 diff（修改前 {old} 字符，修改后 {new} 字符）。',
    requestFailed: '请求失败',
    streamUnavailable: '当前浏览器不支持流式输出。',
    streamFailed: '流式请求失败：{error}',
    rejectedFromUi: '已从网页界面拒绝',
    reviseTryAgain: '请修改后再试。',
    workspaceChanged: '工作目录已切换，并为新目录开启了新会话。',
  },
};

function format(text, values = {}) {
  return Object.entries(values).reduce((result, [key, value]) => result.replace(`{${key}}`, value), text);
}

function approvalProcessingText(approval, t, type) {
  if (type === 'reject' || type === 'feedback') return t.processingApproval;
  const toolName = approval?.toolName;
  if (toolName === 'read_file') return t.processingReadFile;
  if (toolName === 'write_file') return t.processingWriteFile;
  if (toolName === 'edit_file') return t.processingEditFile;
  if (toolName === 'search') return t.processingSearch;
  if (toolName === 'shell') return t.processingShell;
  return format(t.processingTool, { tool: toolName || 'tool' });
}

const LANGUAGE_STORAGE_KEY = 'langcode-language';
const SIDEBAR_STORAGE_KEY = 'langcode-sidebar-collapsed';
const TTS_VOICE_STORAGE_KEY = 'langcode-tts-voice-id';

function getInitialLanguage() {
  return safeStorageGet(LANGUAGE_STORAGE_KEY, '') || 'zh';
}

function getInitialSidebarCollapsed() {
  return safeStorageGet(SIDEBAR_STORAGE_KEY, '') === 'true';
}

const ACTIVE_SESSION_STORAGE_KEY = 'langcode-active-session-id';

function getStoredActiveSessionId() {
  return safeStorageGet(ACTIVE_SESSION_STORAGE_KEY, '') || '';
}

function getInitialTtsVoiceId() {
  return safeStorageGet(TTS_VOICE_STORAGE_KEY, '') || 'default';
}

function getInitialActiveSessionId() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get('sessionId') || params.get('session') || '';
  if (fromQuery) return fromQuery;
  const hash = window.location.hash || '';
  if (hash.startsWith('#session=')) return decodeURIComponent(hash.slice('#session='.length));
  return getStoredActiveSessionId();
}

function writeActiveSessionLocation(sessionId) {
  const url = new URL(window.location.href);
  if (sessionId) {
    url.searchParams.set('sessionId', sessionId);
  } else {
    url.searchParams.delete('sessionId');
  }
  window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
}

function mergeSessions(currentSessionId, sessions, draftSessions) {
  const merged = [];
  const seen = new Set();
  for (const session of [...draftSessions, ...sessions]) {
    if (seen.has(session.id)) continue;
    seen.add(session.id);
    merged.push(session);
  }
  if (currentSessionId && !seen.has(currentSessionId)) {
    merged.unshift({ id: currentSessionId, title: currentSessionId, active: true, draft: true });
  }
  return merged;
}

function groupSessionsByWorkspace(sessions) {
  const groups = [];
  const byWorkspace = new Map();
  for (const session of sessions) {
    const workspace = session.workspace || '';
    if (!byWorkspace.has(workspace)) {
      const group = {
        workspace,
        name: workspaceDisplayName(workspace),
        sessions: [],
      };
      byWorkspace.set(workspace, group);
      groups.push(group);
    }
    byWorkspace.get(workspace).sessions.push(session);
  }
  return groups;
}

function workspaceDisplayName(workspace) {
  if (!workspace) return '未绑定目录';
  const trimmed = String(workspace).replace(/\/+$/, '');
  return trimmed.split('/').filter(Boolean).pop() || trimmed || '/';
}

function modelOptionValue(provider, model, gateway = '') {
  return [provider || 'openai', model || 'glm-5', gateway].filter(Boolean).join(':');
}

function makeModelOption(provider, model, gateway = '') {
  const normalizedProvider = provider || 'openai';
  const normalizedModel = model || 'glm-5';
  const normalizedGateway = gateway || (normalizedProvider === 'openai' && normalizedModel === 'glm-5' ? 'aimp-glm' : '');
  return {
    value: modelOptionValue(normalizedProvider, normalizedModel, normalizedGateway),
    provider: normalizedProvider,
    model: normalizedModel,
    gateway: normalizedGateway,
    label: `${normalizedProvider} ${normalizedModel}`,
  };
}

function findModelOption(value) {
  return MODEL_OPTIONS.find((option) => option.value === value) || null;
}

function isRemovedLegacyModel(provider, model) {
  return provider === 'zhipu' && model === 'glm-5.1';
}

function App() {
  const [language, setLanguage] = useState(getInitialLanguage);
  const t = TRANSLATIONS[language];
  const [sessionId, setSessionId] = useState('');
  const [sessions, setSessions] = useState([]);
  const [draftSessions, setDraftSessions] = useState([]);
  const [sessionMenu, setSessionMenu] = useState(null);
  const [collapsedWorkspaces, setCollapsedWorkspaces] = useState({});
  const [renamingSessionId, setRenamingSessionId] = useState('');
  const [renamingTitle, setRenamingTitle] = useState('');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(getInitialSidebarCollapsed);
  const [status, setStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [slashMenuOpen, setSlashMenuOpen] = useState(false);
  const [slashMenuIndex, setSlashMenuIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [creatingSession, setCreatingSession] = useState(false);
  const [runningSessions, setRunningSessions] = useState({});
  const [pendingApproval, setPendingApproval] = useState(null);
  const [pendingApprovalAssistantId, setPendingApprovalAssistantId] = useState('');
  const [approvalProcessing, setApprovalProcessing] = useState(false);
  const [error, setError] = useState('');
  const [toasts, setToasts] = useState([]);
  const [unreadSessions, setUnreadSessions] = useState({});
  const [themeMode, setThemeMode] = useState(readStoredThemeMode);
  const [approvalEdit, setApprovalEdit] = useState('');
  const lastUserTextRef = useRef('');
  const approvalSectionRef = useRef(null);
  const approvalEditError = useMemo(() => {
    if (!pendingApproval || !approvalEdit.trim()) return '';
    try {
      JSON.parse(approvalEdit);
      return '';
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  }, [approvalEdit, pendingApproval]);
  const [workspaceInput, setWorkspaceInput] = useState('');
  const [selectedWorkspace, setSelectedWorkspace] = useState('');
  const [modelInput, setModelInput] = useState('openai:glm-5:aimp-glm');
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [voiceModel, setVoiceModel] = useState('qwen3-asr-0.6b');
  const [voiceActive, setVoiceActive] = useState(false);
  const [voiceConversationActive, setVoiceConversationActive] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('');
  const [voiceError, setVoiceError] = useState('');
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [ttsVoiceId, setTtsVoiceId] = useState(getInitialTtsVoiceId);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const scrollRef = useRef(null);
  const composerTextareaRef = useRef(null);
  const messagesRef = useRef([]);
  const stickToBottomRef = useRef(true);
  const renameInputRef = useRef(null);
  const renamingSessionIdRef = useRef('');
  const renameSubmittingRef = useRef('');
  const activeSessionRef = useRef('');
  const streamControllersRef = useRef({});
  const activeRunsRef = useRef({});
  const sessionLiveMessagesRef = useRef({});
  const sessionPendingApprovalsRef = useRef({});
  const runningSessionsRef = useRef({});
  const pendingApprovalRef = useRef(null);
  const creatingSessionRef = useRef(false);
  const voiceSocketRef = useRef(null);
  const voiceAudioContextRef = useRef(null);
  const voiceProcessorRef = useRef(null);
  const voiceSourceRef = useRef(null);
  const voiceStreamRef = useRef(null);
  const voiceBufferRef = useRef(new Float32Array(0));
  const voiceStoppingRef = useRef(false);
  const voiceDraftUserIdRef = useRef('');
  const voiceSessionIdRef = useRef('');
  const voiceAutoFinishTimerRef = useRef(null);
  const voiceFinishFallbackTimerRef = useRef(null);
  const voiceAutoFinishTextRef = useRef('');
  const voiceAutoFinishModeRef = useRef('');
  const voiceLastPartialTextRef = useRef('');
  const voiceLastPartialChangedAtRef = useRef(0);
  const ttsAudioRef = useRef(null);
  const ttsPreviewAudioRef = useRef(null);
  const ttsObjectUrlRef = useRef('');
  const ttsPlayingRef = useRef(false);
  const ttsQueueRef = useRef([]);
  const ttsPumpActiveRef = useRef(false);
  const ttsCurrentAbortControllersRef = useRef(new Set());
  const ttsAudioResolveRef = useRef(null);
  const ttsPlaybackTokenRef = useRef('');
  const ttsChunkBuffersRef = useRef({});
  const ttsIncomingTextRef = useRef({});
  const ttsQueuedSpeechRef = useRef({});
  const ttsPlayedSpeechRef = useRef({});
  const ttsPlayedAudioKeysRef = useRef(new Set());
  const ttsSpeechSequenceRef = useRef({});
  const ttsDelayedDisplayRef = useRef({});
  const ttsBargeInTokenRef = useRef('');
  const ttsSuppressedTokensRef = useRef(new Set());
  const ttsDuckedRef = useRef(false);
  const ttsDuckedPositionRef = useRef(0);
  const ttsPlaybackEndedAtRef = useRef(0);
  const voiceSpeechActiveRef = useRef(false);
  const voiceSpeechStartedAtRef = useRef(0);
  const markdownDeltaBuffersRef = useRef({});
  const displayDeltaBuffersRef = useRef({});
  const displayDeltaTimersRef = useRef({});
  const ttsVoiceIdRef = useRef(getInitialTtsVoiceId());
  const voiceActiveRef = useRef(false);
  const voiceConversationActiveRef = useRef(false);
  const voiceRestartTimerRef = useRef(null);
  const bargeInTriggeredRef = useRef(false);
  const bargeInContextRef = useRef(null);
  const toastTimersRef = useRef({});
  const ttsEnabledRef = useRef(true);

  const activeSessionBusy = Boolean(sessionId && runningSessions[sessionId]);
  const interactionBusy = busy || activeSessionBusy;
  const anySessionBusy = Object.keys(runningSessions).length > 0;
  const workspaceBusy = busy || anySessionBusy;
  const visibleSessions = useMemo(
    () => mergeSessions(sessionId, sessions, draftSessions),
    [draftSessions, sessionId, sessions],
  );
  const ttsVoiceOptions = useMemo(
    () => normalizeTtsVoiceOptions(
      status?.tts?.voices?.length
        ? status.tts.voices
        : [
            { id: 'xuefen', name: '雪芬', style: '自定义音色', builtIn: true },
            { id: 'wangju', name: '汪菊', style: '自定义音色', builtIn: true },
          ],
    ),
    [status?.tts?.voices],
  );
  const workspaceGroups = useMemo(() => groupSessionsByWorkspace(visibleSessions), [visibleSessions]);
  const activeWorkspaceLabel = sessionId ? workspaceInput : '';

  const modelOptions = useMemo(() => {
    const current = makeModelOption(status?.provider, status?.model, status?.gateway || '');
    if (isRemovedLegacyModel(current.provider, current.model)) return MODEL_OPTIONS;
    return MODEL_OPTIONS.some((option) => option.value === current.value)
      ? MODEL_OPTIONS
      : [current, ...MODEL_OPTIONS];
  }, [status?.gateway, status?.model, status?.provider]);
  const selectedModelOption = findModelOption(modelInput) || makeModelOption(status?.provider, status?.model);
  const selectedSupportsThinking = Boolean(selectedModelOption.supportsThinking);

  const slashQuery = getSlashQuery(input);
  const filteredSlashCommands = useMemo(() => {
    if (slashQuery === null) return [];
    return SLASH_COMMANDS.filter((item) => item.command.slice(1).startsWith(slashQuery));
  }, [slashQuery]);
  const slashMenuVisible = slashMenuOpen && !pendingApproval && filteredSlashCommands.length > 0;
  const selectedSlashCommand = SLASH_COMMANDS.find(
    (item) => input === item.command || input.startsWith(`${item.command} `),
  );

  useEffect(() => {
    toastSink = pushToast;
    return () => {
      toastSink = null;
    };
  }, []);

  useEffect(() => {
    activeTranslations = t;
  }, [t]);

  useEffect(() => {
    voiceActiveRef.current = voiceActive;
  }, [voiceActive]);

  useEffect(() => {
    voiceConversationActiveRef.current = voiceConversationActive;
  }, [voiceConversationActive]);

  useEffect(() => {
    runningSessionsRef.current = runningSessions;
  }, [runningSessions]);

  useEffect(() => {
    pendingApprovalRef.current = pendingApproval;
  }, [pendingApproval]);

  useEffect(() => {
    ttsVoiceIdRef.current = ttsVoiceId || 'default';
    safeStorageSet(TTS_VOICE_STORAGE_KEY, ttsVoiceIdRef.current);
  }, [ttsVoiceId]);

  useEffect(() => {
    if (!ttsVoiceOptions.length) return;
    if (!ttsVoiceOptions.some((voice) => voice.id === ttsVoiceId)) {
      setTtsVoiceId(ttsVoiceOptions[0].id);
    }
  }, [ttsVoiceOptions, ttsVoiceId]);

  useEffect(() => {
    ttsEnabledRef.current = ttsEnabled;
    if (!ttsEnabled) stopTtsPlayback({ clearQueue: true, stopVoice: true, suppressCurrent: true });
  }, [ttsEnabled]);

  useEffect(() => {
    async function boot() {
      await refreshStatus();
      await refreshSessions({ restoreActive: true });
    }
    void boot();
    return () => {
      stopVoiceCapture({ cancel: true, submit: false });
      for (const token of Object.keys(displayDeltaTimersRef.current)) {
        window.clearTimeout(displayDeltaTimersRef.current[token]);
      }
      displayDeltaTimersRef.current = {};
      displayDeltaBuffersRef.current = {};
    };
  }, []);

  useEffect(() => {
    activeSessionRef.current = sessionId;
    if (sessionId) {
      safeStorageSet(ACTIVE_SESSION_STORAGE_KEY, sessionId);
      writeActiveSessionLocation(sessionId);
    } else {
      safeStorageSet(ACTIVE_SESSION_STORAGE_KEY, null);
      writeActiveSessionLocation('');
    }
  }, [sessionId]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    safeStorageSet(LANGUAGE_STORAGE_KEY, language);
  }, [language]);

  useEffect(() => {
    if (!error) return undefined;
    const timer = window.setTimeout(() => setError(''), 6000);
    return () => window.clearTimeout(timer);
  }, [error]);

  useEffect(() => {
    if (!voiceError) return undefined;
    const timer = window.setTimeout(() => setVoiceError(''), 6000);
    return () => window.clearTimeout(timer);
  }, [voiceError]);

  // One timer per toast id, keyed by id and never rebuilt: the previous version recreated
  // every timer whenever the list changed, so a new toast silently extended the life of all
  // the older ones.
  useEffect(() => {
    const alive = new Set(toasts.map((toast) => toast.id));
    for (const id of Object.keys(toastTimersRef.current)) {
      if (alive.has(id)) continue;
      window.clearTimeout(toastTimersRef.current[id]);
      delete toastTimersRef.current[id];
    }
    for (const toast of toasts) {
      if (toast.fatal || toastTimersRef.current[toast.id]) continue;
      const remaining = Math.max(0, TOAST_TTL_MS - (Date.now() - (toast.createdAt || Date.now())));
      toastTimersRef.current[toast.id] = window.setTimeout(() => {
        delete toastTimersRef.current[toast.id];
        dismissToast(toast.id);
      }, remaining);
    }
  }, [toasts]);

  useEffect(
    () => () => {
      for (const timer of Object.values(toastTimersRef.current)) window.clearTimeout(timer);
      toastTimersRef.current = {};
    },
    [],
  );

  function dismissToast(toastId) {
    setToasts((current) => current.filter((toast) => toast.id !== toastId));
  }

  function pushToast({ message, tone = 'error', fatal = false, retry = null }) {
    const text = String(message || '').trim();
    if (!text) return;
    const now = Date.now();
    setToasts((current) => {
      const duplicate = current.some(
        (toast) => toast.tone === tone && toast.message === text && now - (toast.createdAt || 0) < TOAST_DEDUPE_MS,
      );
      if (duplicate) return current;
      const next = [...current, { id: crypto.randomUUID(), message: text, tone, fatal, retry, createdAt: now }];
      while (next.length > TOAST_LIMIT) {
        // A fatal toast carries the failure the user still has to act on (retry); drop the
        // oldest dismissible one instead, and keep everything when only fatals are left.
        const evictable = next.findIndex((toast) => !toast.fatal);
        if (evictable < 0) break;
        next.splice(evictable, 1);
      }
      return next;
    });
  }

  function applyThemeMode(mode) {
    const nextMode = THEME_MODES.includes(mode) ? mode : 'system';
    setThemeMode(nextMode);
    writeStoredThemeMode(nextMode);
    applyThemeAttribute(nextMode);
    publishThemeChange(isEffectiveDark(nextMode));
  }

  useEffect(() => {
    safeStorageSet(SIDEBAR_STORAGE_KEY, String(sidebarCollapsed));
    if (sidebarCollapsed) {
      setSessionMenu(null);
      setRenamingSessionId('');
    }
  }, [sidebarCollapsed]);

  useEffect(() => {
    if (!stickToBottomRef.current) return;
    scrollToLatest({ behavior: 'auto' });
  }, [messages, pendingApproval]);

  useEffect(() => {
    if (!pendingApproval) return;
    const active = document.activeElement;
    const editing = active instanceof HTMLTextAreaElement || active instanceof HTMLInputElement;
    if (editing) return;
    approvalSectionRef.current?.focus({ preventScroll: true });
  }, [pendingApproval]);

  useEffect(() => {
    resizeComposerTextarea();
  }, [input]);

  useEffect(() => {
    if (!renamingSessionId || !renameInputRef.current) return;
    renameInputRef.current.focus();
    renameInputRef.current.select();
  }, [renamingSessionId]);

  useEffect(() => {
    renamingSessionIdRef.current = renamingSessionId;
  }, [renamingSessionId]);

  useEffect(() => {
    if (!sessionMenu) return undefined;
    const closeMenu = () => setSessionMenu(null);
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') closeMenu();
    };
    const closeOnOutsidePointer = (event) => {
      if (event.target.closest?.('.session-menu, .session-menu-button')) return;
      closeMenu();
    };
    window.addEventListener('pointerdown', closeOnOutsidePointer);
    window.addEventListener('resize', closeMenu);
    window.addEventListener('keydown', closeOnEscape);
    return () => {
      window.removeEventListener('pointerdown', closeOnOutsidePointer);
      window.removeEventListener('resize', closeMenu);
      window.removeEventListener('keydown', closeOnEscape);
    };
  }, [sessionMenu]);

  async function refreshStatus() {
    const response = await api('/api/status');
    setStatus(response);
    setWorkspaceInput(response.workspace || '');
    setSelectedWorkspace(response.workspace || '');
    const nextProvider = response.provider || 'openai';
    const nextModel = response.model || 'glm-5';
    const nextGateway = response.gateway || (nextProvider === 'openai' && nextModel === 'glm-5' ? 'aimp-glm' : '');
    setModelInput(
      isRemovedLegacyModel(nextProvider, nextModel)
        ? 'openai:glm-5:aimp-glm'
        : modelOptionValue(nextProvider, nextModel, nextGateway),
    );
    setThinkingEnabled(Boolean(response.thinking));
  }

  async function refreshSessions({ restoreActive = false } = {}) {
    const response = await api('/api/sessions');
    if (response.ok) {
      const loadedSessions = response.sessions || [];
      setSessions(loadedSessions);
      if (restoreActive && !activeSessionRef.current) {
        const storedSessionId = getInitialActiveSessionId();
        if (storedSessionId && loadedSessions.some((session) => session.id === storedSessionId)) {
          await openSession(storedSessionId);
        } else if (storedSessionId) {
          safeStorageSet(ACTIVE_SESSION_STORAGE_KEY, null);
          writeActiveSessionLocation('');
        }
      }
    }
  }

  function markSessionRunning(targetSessionId, running) {
    setRunningSessions((current) => {
      if (!targetSessionId) return current;
      if (running) return { ...current, [targetSessionId]: true };
      const next = { ...current };
      delete next[targetSessionId];
      return next;
    });
  }

  function clearRunIfCurrent(targetSessionId, runId) {
    const activeRun = activeRunsRef.current[targetSessionId];
    if (!activeRun || activeRun.runId !== runId) return false;
    delete activeRunsRef.current[targetSessionId];
    delete streamControllersRef.current[targetSessionId];
    markSessionRunning(targetSessionId, false);
    return true;
  }

  // The rendered list is authoritative only for the active session; a background session's
  // state lives in the live-message cache.
  function sessionMessagesFor(targetSessionId) {
    if (activeSessionRef.current === targetSessionId) return messagesRef.current;
    return (
      sessionLiveMessagesRef.current[targetSessionId] ||
      activeRunsRef.current[targetSessionId]?.initialMessages ||
      []
    );
  }

  function updateSessionMessages(targetSessionId, updater) {
    if (activeSessionRef.current === targetSessionId) {
      setMessages((current) => {
        const next = updater(current);
        sessionLiveMessagesRef.current[targetSessionId] = next;
        return next;
      });
      return;
    }
    const activeRun = activeRunsRef.current[targetSessionId];
    const current = sessionLiveMessagesRef.current[targetSessionId] || activeRun?.initialMessages || [];
    sessionLiveMessagesRef.current[targetSessionId] = updater(current);
  }

  function upsertVoiceDraftMessage(text) {
    const targetSessionId = voiceSessionIdRef.current || sessionId;
    if (!targetSessionId || activeSessionRef.current !== targetSessionId) return '';
    const content = String(text || '').trim();
    if (!content) return voiceDraftUserIdRef.current;
    let draftId = voiceDraftUserIdRef.current;
    if (!draftId) {
      draftId = crypto.randomUUID();
      voiceDraftUserIdRef.current = draftId;
    }
    stickToBottomRef.current = true;
    setShowJumpToLatest(false);
    const current = messagesRef.current;
    const exists = current.some((message) => message.id === draftId);
    const next = exists
      ? current.map((message) =>
          message.id === draftId ? { ...message, content, voiceDraft: true, kind: 'message', role: 'user' } : message,
        )
      : [...archiveAssistantProgress(current), { id: draftId, role: 'user', kind: 'message', content, voiceDraft: true }];
    messagesRef.current = next;
    sessionLiveMessagesRef.current[targetSessionId] = next;
    setMessages(next);
    return draftId;
  }

  function clearVoiceDraftMessage() {
    const draftId = voiceDraftUserIdRef.current;
    const targetSessionId = voiceSessionIdRef.current || sessionId;
    if (!draftId) return;
    voiceDraftUserIdRef.current = '';
    if (activeSessionRef.current === targetSessionId) {
      setMessages((current) => {
        const next = current.filter((message) => message.id !== draftId);
        messagesRef.current = next;
        sessionLiveMessagesRef.current[targetSessionId] = next;
        return next;
      });
    } else if (targetSessionId) {
      sessionLiveMessagesRef.current[targetSessionId] = (sessionLiveMessagesRef.current[targetSessionId] || []).filter(
        (message) => message.id !== draftId,
      );
    }
  }

  function updateAssistantProgress(assistantId, updater) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== assistantId) return message;
        const progress = message.progress || emptyProgress();
        const nextMessage = { ...message, progress: updater(progress) };
        nextMessage.contentPlacement = contentPlacementForMessage(nextMessage);
        return nextMessage;
      }),
    );
  }

  function setAssistantTodos(assistantId, nextTodos) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== assistantId) return message;
        return {
          ...message,
          todos: nextTodos,
          contentPlacement: contentPlacementForMessage({ ...message, todos: nextTodos }),
        };
      }),
    );
  }

  function setAssistantProgressRunning(assistantId, running) {
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? { ...message, progressRunning: running, thinkingRunning: running ? message.thinkingRunning : false }
          : message,
      ),
    );
  }

  function updateLatestAssistantProgress(updater) {
    setMessages((current) => {
      const index = findLastAssistantIndex(current);
      if (index < 0) return current;
      const next = [...current];
      const message = next[index];
      const nextProgress = updater(message.progress || emptyProgress());
      next[index] = {
        ...message,
        progress: nextProgress,
        contentPlacement: contentPlacementForMessage({ ...message, progress: nextProgress }),
      };
      return next;
    });
  }

  function updateLatestAssistantTodos(nextTodos) {
    setMessages((current) => attachTodosToLastAssistant(current, nextTodos));
  }

  async function sendMessage(event) {
    event?.preventDefault();
    await submitMessage(input, { clearInput: true });
  }

  async function submitMessage(
    rawText,
    {
      clearInput = false,
      reuseUserMessageId = '',
      ignoreStaleRunning = false,
      voiceInterrupt = null,
      targetSessionId = '',
      dropAssistantMessageId = '',
    } = {},
  ) {
    const text = String(rawText || '').trim();
    const activeSessionId = targetSessionId || sessionId;
    if (!activeSessionId) {
      setError(t.chooseWorkspaceFirst);
      return;
    }
    if (!text || busy || activeRunsRef.current[activeSessionId] || (!ignoreStaleRunning && runningSessions[activeSessionId])) return;
    lastUserTextRef.current = text;
    stickToBottomRef.current = true;
    setShowJumpToLatest(false);
    stopTtsPlayback({ clearQueue: true, stopVoice: !voiceConversationActiveRef.current, suppressCurrent: true });
    if (clearInput) setInput('');
    setSlashMenuOpen(false);
    setError('');
    const assistantId = crypto.randomUUID();
    const runId = crypto.randomUUID();
    const visibleMessages = sessionMessagesFor(activeSessionId).filter(
      (message) =>
        !(
          dropAssistantMessageId &&
          message.id === dropAssistantMessageId &&
          message.role === 'assistant' &&
          !String(message.content || '').trim()
        ),
    );
    const archivedMessages = archiveAssistantProgress(visibleMessages);
    const hasReusableUserMessage = Boolean(
      reuseUserMessageId && archivedMessages.some((message) => message.id === reuseUserMessageId && message.role === 'user'),
    );
    const userMessages = hasReusableUserMessage
      ? archivedMessages.map((message) =>
          message.id === reuseUserMessageId
            ? { ...message, kind: 'message', content: text, voiceDraft: false }
            : message,
        )
      : [...archivedMessages, { id: crypto.randomUUID(), role: 'user', kind: 'message', content: text }];
    const userMessageId = hasReusableUserMessage ? reuseUserMessageId : userMessages[userMessages.length - 1]?.id || '';
    const initialMessages = [
      ...userMessages,
      {
        id: assistantId,
        role: 'assistant',
        kind: 'message',
        content: '',
        thinking: '',
        thinkingRunning: Boolean(selectedSupportsThinking && thinkingEnabled),
        progress: emptyProgress(),
        todos: [],
        progressRunning: true,
      },
    ];
    activeRunsRef.current[activeSessionId] = { runId, assistantId, userMessageId, text, controller: null, initialMessages };
    sessionLiveMessagesRef.current[activeSessionId] = initialMessages;
    if (activeSessionRef.current === activeSessionId) {
      messagesRef.current = initialMessages;
      setMessages(initialMessages);
    }
    markSessionRunning(activeSessionId, true);
    await streamChat(
      { sessionId: activeSessionId, message: text, voiceInterrupt, voiceMode: voiceConversationActiveRef.current },
      assistantId,
      activeSessionId,
      runId,
    );
    if (activeRunsRef.current[activeSessionId]?.runId === runId) {
      setAssistantProgressRunning(assistantId, false);
      clearRunIfCurrent(activeSessionId, runId);
    }
    refreshSessions();
  }

  function stopGeneration(options = {}) {
    const { stopVoice = true } = options;
    const activeSessionId = sessionId;
    const activeRun = activeRunsRef.current[activeSessionId];
    if (!activeSessionId || !activeRun) return;
    stopTtsPlayback({
      clearQueue: true,
      stopVoice,
      suppressCurrent: true,
      suppressToken: ttsToken(activeSessionId, activeRun.assistantId),
    });
    void api('/api/session/cancel', { sessionId: activeSessionId, runId: activeRun.runId }).catch(() => {});
    activeRun.controller?.abort();
    streamControllersRef.current[activeSessionId]?.abort();
    const cancelledAssistantId = activeRun.responseAssistantId || activeRun.assistantId;
    if (shouldDiscardInterruptedRun(messagesRef.current, activeRun)) {
      if (cancelledAssistantId) clearMarkdownDeltaBuffer(activeSessionId, cancelledAssistantId, { flush: false });
      const nextMessages = messagesRef.current.filter(
        (message) => message.id !== activeRun.assistantId && message.id !== activeRun.userMessageId,
      );
      messagesRef.current = nextMessages;
      sessionLiveMessagesRef.current[activeSessionId] = nextMessages;
      setMessages(nextMessages);
      clearRunIfCurrent(activeSessionId, activeRun.runId);
      refreshSessions();
      return;
    }
    if (cancelledAssistantId) clearMarkdownDeltaBuffer(activeSessionId, cancelledAssistantId);
    updateLatestAssistantProgress((current) => finishAgentProgress(current, t));
    const assistantId = activeRun.assistantId || findLastAssistantId(messages);
    if (assistantId) setAssistantProgressRunning(assistantId, false);
    clearRunIfCurrent(activeSessionId, activeRun.runId);
  }

  function forceEndConversation() {
    const activeSessionId = sessionId;
    if (!activeSessionId) return;
    const activeRun = activeRunsRef.current[activeSessionId];
    if (activeRun?.assistantId) clearMarkdownDeltaBuffer(activeSessionId, activeRun.assistantId);
    stopTtsPlayback({
      clearQueue: true,
      stopVoice: true,
      suppressCurrent: true,
      suppressToken: activeRun?.assistantId ? ttsToken(activeSessionId, activeRun.assistantId) : '',
    });
    resetVoiceCaptureState({ clearDraft: true });
    if (activeRun?.runId) {
      void api('/api/session/cancel', { sessionId: activeSessionId, runId: activeRun.runId }).catch(() => {});
      activeRun.controller?.abort();
      streamControllersRef.current[activeSessionId]?.abort();
      const assistantId = activeRun.assistantId || findLastAssistantId(messagesRef.current);
      if (assistantId) {
        updateSessionMessages(activeSessionId, (current) =>
          setAssistantProgressRunningInMessages(
            updateAssistantProgressInMessages(current, assistantId, (progress) => finishAgentProgress(progress, t)),
            assistantId,
            false,
          ).map((message) => (message.id === assistantId ? { ...message, thinkingRunning: false } : message)),
        );
      }
      clearRunIfCurrent(activeSessionId, activeRun.runId);
    } else {
      markSessionRunning(activeSessionId, false);
    }
    delete sessionPendingApprovalsRef.current[activeSessionId];
    setPendingApproval(null);
    setPendingApprovalAssistantId('');
    setApprovalProcessing(false);
    setApprovalEdit('');
    setError('');
    setSlashMenuOpen(false);
    sessionLiveMessagesRef.current[activeSessionId] = (sessionLiveMessagesRef.current[activeSessionId] || messagesRef.current).map(
      (message) => (message.role === 'assistant' ? { ...message, progressRunning: false, thinkingRunning: false } : message),
    );
    if (activeSessionRef.current === activeSessionId) {
      setMessages(sessionLiveMessagesRef.current[activeSessionId]);
    }
    void refreshSessions();
  }

  function stopCurrentTtsAudio() {
    const resolveCurrentAudio = ttsAudioResolveRef.current;
    ttsAudioResolveRef.current = null;
    if (ttsAudioRef.current) {
      try {
        ttsAudioRef.current.pause();
        ttsAudioRef.current.currentTime = 0;
      } catch {
        // Audio interruption is best effort; the next playback recreates the element.
      }
    }
    if (ttsObjectUrlRef.current) {
      URL.revokeObjectURL(ttsObjectUrlRef.current);
      ttsObjectUrlRef.current = '';
    }
    ttsAudioRef.current = null;
    if (resolveCurrentAudio) resolveCurrentAudio();
  }

  function currentTtsToken() {
    return ttsPlaybackTokenRef.current || ttsBargeInTokenRef.current || '';
  }

  function stopTtsPlayback({ clearQueue = true, stopVoice = true, suppressCurrent = false, suppressToken = '' } = {}) {
    const tokensToSuppress = new Set();
    if (suppressToken) tokensToSuppress.add(suppressToken);
    if (suppressCurrent) {
      const currentToken = currentTtsToken();
      if (currentToken) tokensToSuppress.add(currentToken);
    }
    for (const token of tokensToSuppress) {
      ttsSuppressedTokensRef.current.add(token);
      requestTtsTurnCancel(token);
      noteVoiceMetricsStop(token, 'stopped');
    }
    clearVoiceTtsPreparingForStoppedTokens(tokensToSuppress);
    restoreTtsPlaybackVolume();
    ttsPlayingRef.current = false;
    if (clearQueue) {
      ttsQueueRef.current = [];
      ttsChunkBuffersRef.current = {};
      if (tokensToSuppress.size > 0) {
        for (const token of tokensToSuppress) {
          delete ttsIncomingTextRef.current[token];
          delete ttsQueuedSpeechRef.current[token];
          delete ttsPlayedSpeechRef.current[token];
          delete ttsSpeechSequenceRef.current[token];
          delete ttsDelayedDisplayRef.current[token];
          delete markdownDeltaBuffersRef.current[token];
          forgetTtsPlayedAudioKeys(token);
        }
      } else {
        ttsIncomingTextRef.current = {};
        ttsQueuedSpeechRef.current = {};
        ttsPlayedSpeechRef.current = {};
        ttsSpeechSequenceRef.current = {};
        ttsDelayedDisplayRef.current = {};
        markdownDeltaBuffersRef.current = {};
        ttsPlayedAudioKeysRef.current.clear();
      }
      ttsPlaybackTokenRef.current = '';
      ttsBargeInTokenRef.current = '';
      if (tokensToSuppress.size === 0) ttsSuppressedTokensRef.current.clear();
    }
    if (ttsCurrentAbortControllersRef.current.size > 0) {
      for (const controller of ttsCurrentAbortControllersRef.current) {
        try {
          controller.abort();
        } catch {
          // The controller may already be closed by the fetch path.
        }
      }
      ttsCurrentAbortControllersRef.current.clear();
    }
    stopCurrentTtsAudio();
    if (stopVoice && !voiceConversationActiveRef.current && (voiceActiveRef.current || voiceSocketRef.current)) {
      void stopVoiceCapture({ cancel: true, submit: false });
    }
  }

  function queueAssistantTtsDelta(targetSessionId, assistantId, delta, { force = false, delayDisplay = false } = {}) {
    // Every early return below means the text will never be spoken. With delayDisplay the
    // caller handed us the only copy of the text, so it must be shown here, otherwise an
    // approval mid-stream silently swallows the rest of the answer and the spinner hangs.
    // Without delayDisplay the caller already displayed it, so re-displaying would duplicate.
    const fallbackToDisplay = () => {
      if (!targetSessionId || !assistantId) return;
      if (delayDisplay) {
        queueDisplayDelta(targetSessionId, assistantId, delta || '');
        if (force) flushQueuedDisplayDelta(targetSessionId, assistantId, { flushMarkdown: true });
      }
      setAssistantVoiceTtsPreparing(targetSessionId, assistantId, false);
    };
    if (
      !ttsEnabledRef.current ||
      !voiceConversationActiveRef.current ||
      !targetSessionId ||
      !assistantId ||
      activeSessionRef.current !== targetSessionId
    ) {
      fallbackToDisplay();
      return;
    }
    if (pendingApprovalRef.current || sessionPendingApprovalsRef.current[targetSessionId]) {
      fallbackToDisplay();
      return;
    }
    const token = ttsToken(targetSessionId, assistantId);
    if (ttsSuppressedTokensRef.current.has(token)) {
      fallbackToDisplay();
      return;
    }
    if (ttsPlaybackTokenRef.current && ttsPlaybackTokenRef.current !== token) {
      stopTtsPlayback({ clearQueue: true, stopVoice: true, suppressCurrent: true });
    }
    if (!ttsPlaybackTokenRef.current) ttsPlaybackTokenRef.current = token;
    refreshTtsBargeInContext(targetSessionId, assistantId);
    const selectedVoiceId = ttsVoiceIdRef.current || 'default';
    const useCustomVoice = selectedVoiceId !== 'default';
    const shouldDelayDisplay = Boolean(delayDisplay && useCustomVoice);
    if (delayDisplay && !useCustomVoice) {
      // The voice switched back to the built-in one mid-stream: the delayed-display queue
      // will not carry this text, so show it now. It is still spoken below.
      queueDisplayDelta(targetSessionId, assistantId, delta || '');
      setAssistantVoiceTtsPreparing(targetSessionId, assistantId, false);
    }
    if (shouldDelayDisplay && !ttsDelayedDisplayRef.current[token]) {
      ttsDelayedDisplayRef.current[token] = { released: false };
    }
    const incomingDelta = delta ? takeNovelTtsIncomingDelta(token, delta) : '';
    const current = `${ttsChunkBuffersRef.current[token] || ''}${incomingDelta}`;
    const { chunks, remainder } = splitTtsText(current, {
      force,
      compact: useCustomVoice,
      first: !ttsSpeechSequenceRef.current[token],
    });
    ttsChunkBuffersRef.current[token] = remainder;
    for (const chunk of chunks) {
      const text = dedupeTtsSpeechChunk(token, stripMarkdownForSpeech(chunk));
      if (!text) {
        if (shouldDelayDisplay) appendDelayedAssistantContent(targetSessionId, assistantId, chunk);
        continue;
      }
      const normalizedText = normalizeTtsSpeechForDedupe(text);
      const liveMessages = sessionLiveMessagesRef.current[targetSessionId] || messagesRef.current;
      const previousUser = findPreviousUserMessage(liveMessages, assistantId);
      const speechId = nextTtsSpeechId(token);
      const queuedAt = Date.now();
      recordVoiceMetric(token, 'queuedAt', queuedAt, { sessionId: targetSessionId });
      ttsQueueRef.current.push({
        type: 'speech',
        speechId,
        // The token is per assistant reply, so it doubles as the backend turn id and the
        // sequence counter behind speechId gives the 0-based segment index.
        segmentIndex: (ttsSpeechSequenceRef.current[token] || 1) - 1,
        token,
        sessionId: targetSessionId,
        assistantId,
        voiceId: selectedVoiceId,
        userText: previousUser?.content || '',
        text,
        normalizedText,
        displayText: shouldDelayDisplay ? chunk : '',
        delayDisplay: shouldDelayDisplay,
        queuedAt,
      });
    }
    if (force) {
      ttsQueueRef.current.push({ type: 'done', token, sessionId: targetSessionId, assistantId, delayDisplay: shouldDelayDisplay });
      delete ttsChunkBuffersRef.current[token];
    }
    void pumpTtsQueue();
  }

  function dedupeTtsSpeechChunk(token, text) {
    const original = String(text || '').trim();
    if (!original) return '';
    const state = ttsQueuedSpeechRef.current[token] || { last: '', spoken: '' };
    const normalized = normalizeTtsSpeechForDedupe(original);
    const spoken = state.spoken || normalizeTtsSpeechForDedupe(state.last);
    const previous = normalizeTtsSpeechForDedupe(state.last);
    let nextText = original;
    if (spoken && normalized && spoken.includes(normalized)) return '';
    if (spoken && normalized.startsWith(spoken)) {
      nextText = stripNormalizedTtsPrefix(original, spoken.length).trimStart();
    } else {
      const cumulativePrefix = longestTtsPrefixAlreadySpoken(spoken, normalized);
      if (cumulativePrefix >= 8) nextText = stripNormalizedTtsPrefix(original, cumulativePrefix).trimStart();
    }
    if (nextText !== original) {
      // The model or stream replayed an already spoken prefix; only enqueue the novel tail.
    } else if (previous && normalized === previous) {
      return '';
    } else if (previous && original.startsWith(state.last)) {
      nextText = original.slice(state.last.length).trimStart();
    } else if (previous && normalized.startsWith(previous)) {
      nextText = stripNormalizedTtsPrefix(original, previous.length).trimStart();
    } else {
      const overlap = longestTtsTextOverlap(previous, normalized);
      if (overlap >= 6) nextText = stripNormalizedTtsPrefix(original, overlap).trimStart();
    }
    nextText = cleanTtsNovelText(nextText);
    if (!nextText) return '';
    const nextNormalized = normalizeTtsSpeechForDedupe(nextText);
    if (spoken && nextNormalized && spoken.includes(nextNormalized)) return '';
    ttsQueuedSpeechRef.current[token] = {
      last: nextText,
      spoken: appendNormalizedTtsSpeech(spoken, nextNormalized),
    };
    return nextText;
  }

  function nextTtsSpeechId(token) {
    const next = (ttsSpeechSequenceRef.current[token] || 0) + 1;
    ttsSpeechSequenceRef.current[token] = next;
    return `${token}:${next}`;
  }

  function takeNovelTtsIncomingDelta(token, delta) {
    const value = String(delta || '');
    if (!value) return '';
    const state = ttsIncomingTextRef.current[token] || { text: '' };
    let novel = value;
    if (state.text && value.length > state.text.length && value.startsWith(state.text)) {
      novel = value.slice(state.text.length);
      state.text = value;
    } else {
      state.text = `${state.text || ''}${value}`;
    }
    ttsIncomingTextRef.current[token] = state;
    return novel;
  }

  function normalizeTtsSpeechForDedupe(text) {
    return String(text || '')
      .replace(/\s+/g, '')
      .replace(/[，,。！？!?；;：:、"'“”‘’（）()[\]【】]/g, '')
      .trim();
  }

  function longestTtsTextOverlap(previous, current) {
    const max = Math.min(previous.length, current.length, 48);
    for (let length = max; length >= 6; length -= 1) {
      if (previous.slice(-length) === current.slice(0, length)) return length;
    }
    return 0;
  }

  function appendNormalizedTtsSpeech(spoken, next) {
    if (!next) return spoken || '';
    if (!spoken) return next;
    if (spoken.includes(next)) return spoken;
    const overlap = longestTtsTextOverlap(spoken, next);
    return `${spoken}${next.slice(overlap)}`;
  }

  function cleanTtsNovelText(text) {
    return String(text || '').replace(/^[\s，,。！？!?；;：:、]+/, '').trim();
  }

  function longestTtsPrefixAlreadySpoken(spoken, current) {
    if (!spoken || !current) return 0;
    const max = Math.min(current.length, 80);
    for (let length = max; length >= 8; length -= 1) {
      if (spoken.includes(current.slice(0, length))) return length;
    }
    return 0;
  }

  function stripNormalizedTtsPrefix(text, normalizedLength) {
    if (normalizedLength <= 0) return text;
    let consumed = 0;
    for (let index = 0; index < text.length; index += 1) {
      const normalizedChar = normalizeTtsSpeechForDedupe(text[index]);
      if (normalizedChar) consumed += normalizedChar.length;
      if (consumed >= normalizedLength) return text.slice(index + 1);
    }
    return '';
  }

  function refreshTtsBargeInContext(targetSessionId, assistantId, userText = '') {
    const liveMessages = sessionLiveMessagesRef.current[targetSessionId] || messagesRef.current;
    const assistantMessage = liveMessages.find((item) => item.id === assistantId);
    const previousUser = findPreviousUserMessage(liveMessages, assistantId);
    bargeInContextRef.current = {
      userText: userText || previousUser?.content || '',
      assistantText: assistantMessage?.content || '',
    };
  }

  // Delaying display is only safe while TTS can actually consume the delta. Once the token
  // is suppressed (an approval stopped playback) or the session is waiting for an approval,
  // queueAssistantTtsDelta drops everything it is handed, so display must not be delayed.
  function shouldDelayCustomVoiceDisplay(targetSessionId, assistantId = '') {
    if (!ttsEnabledRef.current) return false;
    if (!voiceConversationActiveRef.current) return false;
    if (activeSessionRef.current !== targetSessionId) return false;
    if ((ttsVoiceIdRef.current || 'default') === 'default') return false;
    if (pendingApprovalRef.current || sessionPendingApprovalsRef.current[targetSessionId]) return false;
    const token = assistantId ? ttsToken(targetSessionId, assistantId) : '';
    if (token && ttsSuppressedTokensRef.current.has(token)) return false;
    return true;
  }

  function setAssistantVoiceTtsPreparing(targetSessionId, assistantId, preparing) {
    updateSessionMessages(targetSessionId, (current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              voiceTtsPreparing: Boolean(preparing),
              thinkingRunning: preparing ? false : message.thinkingRunning,
            }
          : message,
      ),
    );
  }

  function clearVoiceTtsPreparingForStoppedTokens(tokens) {
    let clearedSpecificMessage = false;
    for (const token of tokens || []) {
      const parsed = parseTtsToken(token);
      if (!parsed) continue;
      clearedSpecificMessage = true;
      setAssistantVoiceTtsPreparing(parsed.sessionId, parsed.assistantId, false);
    }
    if (!clearedSpecificMessage && activeSessionRef.current) {
      clearVoiceTtsPreparingInSession(activeSessionRef.current);
    }
  }

  function clearVoiceTtsPreparingInSession(targetSessionId) {
    if (!targetSessionId) return;
    updateSessionMessages(targetSessionId, (current) =>
      current.map((message) =>
        message.role === 'assistant' && message.voiceTtsPreparing ? { ...message, voiceTtsPreparing: false } : message,
      ),
    );
  }

  function appendDelayedAssistantContent(targetSessionId, assistantId, content) {
    const value = takeMarkdownDisplayDelta(targetSessionId, assistantId, content);
    if (!value) return;
    updateSessionMessages(targetSessionId, (current) =>
      current.map((message) => {
        if (message.id !== assistantId) return message;
        const nextContent = `${message.content || ''}${value}`;
        return {
          ...message,
          content: nextContent,
          voiceTtsPreparing: false,
          thinkingRunning: false,
          contentPlacement: contentPlacementForMessage({ ...message, content: nextContent }),
        };
      }),
    );
  }

  function markdownDeltaToken(targetSessionId, assistantId) {
    return `${targetSessionId}:${assistantId}`;
  }

  function takeMarkdownDisplayDelta(targetSessionId, assistantId, delta, { flush = false } = {}) {
    const token = markdownDeltaToken(targetSessionId, assistantId);
    const combined = `${markdownDeltaBuffersRef.current[token] || ''}${delta || ''}`;
    if (!combined) return '';
    if (flush) {
      delete markdownDeltaBuffersRef.current[token];
      return combined;
    }
    const lastNewline = combined.lastIndexOf('\n');
    const tail = lastNewline >= 0 ? combined.slice(lastNewline + 1) : combined;
    if (tail && isLikelyStreamingMarkdownTableRow(tail)) {
      markdownDeltaBuffersRef.current[token] = tail;
      return lastNewline >= 0 ? combined.slice(0, lastNewline + 1) : '';
    }
    delete markdownDeltaBuffersRef.current[token];
    return combined;
  }

  function flushMarkdownDisplayDelta(targetSessionId, assistantId) {
    return takeMarkdownDisplayDelta(targetSessionId, assistantId, '', { flush: true });
  }

  function cancelDisplayDeltaTimer(token) {
    const timer = displayDeltaTimersRef.current[token];
    if (timer) window.clearTimeout(timer);
    delete displayDeltaTimersRef.current[token];
  }

  function flushQueuedDisplayDelta(targetSessionId, assistantId, { flushMarkdown = false } = {}) {
    const token = markdownDeltaToken(targetSessionId, assistantId);
    cancelDisplayDeltaTimer(token);
    const pending = displayDeltaBuffersRef.current[token] || '';
    delete displayDeltaBuffersRef.current[token];
    const value = takeMarkdownDisplayDelta(targetSessionId, assistantId, pending, { flush: flushMarkdown });
    if (!value) return;
    updateSessionMessages(targetSessionId, (current) =>
      appendAssistantContentInMessages(current, assistantId, value),
    );
  }

  // setTimeout instead of requestAnimationFrame: rAF is throttled to a stop in background
  // tabs, which froze streaming text until the run finished.
  function queueDisplayDelta(targetSessionId, assistantId, delta) {
    if (!delta) return;
    const token = markdownDeltaToken(targetSessionId, assistantId);
    displayDeltaBuffersRef.current[token] = `${displayDeltaBuffersRef.current[token] || ''}${delta}`;
    if (displayDeltaTimersRef.current[token]) return;
    displayDeltaTimersRef.current[token] = window.setTimeout(() => {
      delete displayDeltaTimersRef.current[token];
      flushQueuedDisplayDelta(targetSessionId, assistantId);
    }, DISPLAY_DELTA_FLUSH_MS);
  }

  function clearMarkdownDeltaBuffer(targetSessionId, assistantId, { flush = true } = {}) {
    if (flush) flushQueuedDisplayDelta(targetSessionId, assistantId, { flushMarkdown: true });
    const token = markdownDeltaToken(targetSessionId, assistantId);
    cancelDisplayDeltaTimer(token);
    delete displayDeltaBuffersRef.current[token];
    delete markdownDeltaBuffersRef.current[token];
  }

  function flushActiveRunDisplayDelta(targetSessionId) {
    if (!targetSessionId) return;
    const activeRun = activeRunsRef.current[targetSessionId];
    const assistantId = activeRun?.responseAssistantId || activeRun?.assistantId;
    if (!assistantId) return;
    flushQueuedDisplayDelta(targetSessionId, assistantId);
  }

  function ensureTtsBargeInListening(item) {
    if (
      !item?.token ||
      item.token !== ttsPlaybackTokenRef.current ||
      ttsSuppressedTokensRef.current.has(item.token) ||
      activeSessionRef.current !== item.sessionId ||
      !voiceConversationActiveRef.current ||
      voiceActiveRef.current
    ) return;
    refreshTtsBargeInContext(item.sessionId, item.assistantId, item.userText);
    ttsBargeInTokenRef.current = item.token;
    void startVoiceCapture({ bargeIn: true, context: bargeInContextRef.current });
  }

  function currentBargeInContext() {
    const activeSessionId = sessionId;
    const activeRun = activeRunsRef.current[activeSessionId];
    const liveMessages = sessionLiveMessagesRef.current[activeSessionId] || messagesRef.current;
    const assistantMessage =
      liveMessages.find((item) => item.id === activeRun?.assistantId) ||
      [...liveMessages].reverse().find((item) => item.role === 'assistant');
    const previousUser = assistantMessage
      ? findPreviousUserMessage(liveMessages, assistantMessage.id)
      : [...liveMessages].reverse().find((item) => item.role === 'user');
    return {
      userText: previousUser?.content || '',
      assistantText: assistantMessage?.content || '',
    };
  }

  async function pumpTtsQueue() {
    if (ttsPumpActiveRef.current) return;
    ttsPumpActiveRef.current = true;
    const inFlight = [];
    try {
      while (true) {
        fillTtsInFlight(inFlight);

        if (inFlight.length > 0) {
          const first = inFlight[0];
          if (shouldHoldDelayedTtsPlayback(first.item, inFlight)) {
            await sleep(80);
            continue;
          }
          if (first.item.delayDisplay && !ttsDelayedDisplayRef.current[first.item.token]?.released && inFlight.length >= 2) {
            await prepareTtsEntry(first);
            await waitForPreparedTtsEntry(inFlight[1], 450);
            if (ttsDelayedDisplayRef.current[first.item.token]) {
              ttsDelayedDisplayRef.current[first.item.token].released = true;
            }
          }
          const next = inFlight.shift();
          const prepared = await prepareTtsEntry(next);
          // Refill before awaiting playback so synthesis of N+1/N+2 overlaps playback of N.
          fillTtsInFlight(inFlight);
          if (!isCurrentTtsSpeechItem(next.item)) {
            releasePreparedTts(prepared);
            continue;
          }
          if (prepared.audioItems.length === 0) {
            releaseDelayedDisplayOnTtsFailure(prepared.item);
            releasePreparedTts(prepared);
            continue;
          }
          await playPreparedTtsItem(prepared);
          continue;
        }

        const item = takeNextTtsQueueItem();
        if (!item) break;
        if (item.type === 'done') {
          await finishTtsTokenIfIdle(item);
          continue;
        }
      }
    } finally {
      // The loop only exits with an empty list; on a thrown error the prefetched streams
      // would otherwise keep their fetch and blob URLs alive.
      for (const entry of inFlight) releasePreparedTts(entry.stream);
      inFlight.length = 0;
      ttsPumpActiveRef.current = false;
      if (ttsQueueRef.current.length > 0) void pumpTtsQueue();
    }
  }

  // Bounded by the prefetch window, so the in-flight list can never grow unbounded.
  function fillTtsInFlight(inFlight) {
    while (inFlight.length < ttsPrefetchWindow()) {
      const nextItem = takeNextTtsQueueItem({ speechOnly: true });
      if (!nextItem) return;
      inFlight.push({ item: nextItem, stream: prepareTtsStreamItem(nextItem), prepared: null });
    }
  }

  async function prepareTtsEntry(entry) {
    if (!entry.prepared) entry.prepared = await waitForFirstTtsStreamAudio(entry.stream);
    return entry.prepared;
  }

  function shouldHoldDelayedTtsPlayback(item, inFlight) {
    if (!item?.delayDisplay) return false;
    if (ttsSuppressedTokensRef.current.has(item.token)) return false;
    const state = ttsDelayedDisplayRef.current[item.token];
    if (!state || state.released) return false;
    if (inFlight.length >= 2) return false;
    if (Date.now() - (item.queuedAt || 0) > 900) return false;
    return !hasQueuedTtsDone(item.token);
  }

  async function waitForPreparedTtsEntry(entry, timeoutMs) {
    if (!entry || entry.prepared) return entry?.prepared || null;
    const timeout = new Promise((resolve) => window.setTimeout(() => resolve(null), timeoutMs));
    return Promise.race([prepareTtsEntry(entry), timeout]);
  }

  function hasQueuedTtsDone(token) {
    return ttsQueueRef.current.some((item) => item?.token === token && item.type === 'done');
  }

  function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  // Latency instrumentation only: nothing here feeds the UI.
  function voiceMetricsStore() {
    if (!window.__langcodeVoiceMetrics) window.__langcodeVoiceMetrics = { turns: {}, order: [] };
    return window.__langcodeVoiceMetrics;
  }

  function voiceMetricsTurn(turnId, sessionId = '') {
    const store = voiceMetricsStore();
    if (!store.turns[turnId]) {
      store.turns[turnId] = {
        turnId,
        sessionId,
        queuedAt: 0,
        firstFetchAt: 0,
        firstAudioAt: 0,
        firstPlayAt: 0,
        firstAudioMs: 0,
        gapMs: [],
        bargeInDetectedAt: 0,
        playbackStoppedAt: 0,
      };
      store.order.push(turnId);
      while (store.order.length > VOICE_METRICS_TURN_LIMIT) {
        delete store.turns[store.order.shift()];
      }
    }
    return store.turns[turnId];
  }

  function recordVoiceMetric(turnId, key, value, extra = {}) {
    if (!turnId) return;
    const turn = voiceMetricsTurn(turnId, extra.sessionId || '');
    if (!turn[key]) turn[key] = value;
    if (extra.firstAudioMs && !turn.firstAudioMs) turn.firstAudioMs = extra.firstAudioMs;
  }

  function recordTtsPlaybackStartMetrics(turnId, sessionId) {
    if (!turnId) return;
    const turn = voiceMetricsTurn(turnId, sessionId);
    const now = Date.now();
    if (!turn.firstPlayAt) turn.firstPlayAt = now;
    else if (ttsPlaybackEndedAtRef.current) turn.gapMs.push(now - ttsPlaybackEndedAtRef.current);
  }

  // Only touches turns that actually reached the TTS queue, so the routine stop paths do not
  // fabricate empty metric entries.
  function noteVoiceMetricsStop(turnId, reason) {
    const turn = voiceMetricsStore().turns[turnId];
    if (!turn) return;
    if (!turn.playbackStoppedAt) turn.playbackStoppedAt = Date.now();
    flushVoiceMetrics(turnId, reason);
  }

  function flushVoiceMetrics(turnId, reason) {
    if (!turnId) return;
    const store = voiceMetricsStore();
    const turn = store.turns[turnId];
    if (!turn || turn.reported) return;
    turn.reported = reason;
    const since = (value) => (value && turn.queuedAt ? value - turn.queuedAt : 0);
    console.debug('[voice]', {
      turn: turnId,
      reason,
      queueToFetchMs: since(turn.firstFetchAt),
      queueToFirstAudioMs: since(turn.firstAudioAt),
      queueToFirstPlayMs: since(turn.firstPlayAt),
      serverFirstAudioMs: turn.firstAudioMs,
      gapMs: turn.gapMs,
      duckedAtSec: turn.duckedAtSec || 0,
      bargeInAfterMs: since(turn.bargeInDetectedAt),
      stoppedAfterMs: since(turn.playbackStoppedAt),
    });
  }

  function ttsPrefetchWindow() {
    const voiceId = ttsVoiceIdRef.current || 'default';
    return voiceId && voiceId !== 'default' ? 3 : 2;
  }

  async function finishTtsTokenIfIdle(item) {
    if (item.token !== ttsPlaybackTokenRef.current || ttsQueueRef.current.length > 0) return;
    const flushedMarkdown = flushMarkdownDisplayDelta(item.sessionId, item.assistantId);
    if (flushedMarkdown) {
      updateSessionMessages(item.sessionId, (current) =>
        appendAssistantContentInMessages(current, item.assistantId, flushedMarkdown),
      );
    }
    setAssistantVoiceTtsPreparing(item.sessionId, item.assistantId, false);
    delete ttsDelayedDisplayRef.current[item.token];
    delete ttsIncomingTextRef.current[item.token];
    delete ttsQueuedSpeechRef.current[item.token];
    delete ttsPlayedSpeechRef.current[item.token];
    delete ttsSpeechSequenceRef.current[item.token];
    delete markdownDeltaBuffersRef.current[item.token];
    forgetTtsPlayedAudioKeys(item.token);
    flushVoiceMetrics(item.token, 'completed');
    ttsPlaybackTokenRef.current = '';
    ttsBargeInTokenRef.current = '';
    if (voiceConversationActiveRef.current) {
      if (voiceActiveRef.current) await stopVoiceCapture({ cancel: true, submit: false });
      scheduleVoiceConversationListen(350);
    } else if (voiceActiveRef.current) {
      await stopVoiceCapture({ cancel: true, submit: false });
    }
  }

  function isCurrentTtsSpeechItem(item) {
    return Boolean(
      item &&
      item.type !== 'done' &&
      item.token === ttsPlaybackTokenRef.current &&
      !ttsSuppressedTokensRef.current.has(item.token) &&
      activeSessionRef.current === item.sessionId
    );
  }

  function takeNextTtsQueueItem({ speechOnly = false } = {}) {
    while (ttsQueueRef.current.length > 0) {
      const item = ttsQueueRef.current.shift();
      if (!item || (item.token && item.token !== ttsPlaybackTokenRef.current)) continue;
      if (item.token && ttsSuppressedTokensRef.current.has(item.token)) continue;
      if (speechOnly && item.type === 'done') {
        ttsQueueRef.current.unshift(item);
        return null;
      }
      if (item.type !== 'done' && activeSessionRef.current !== item.sessionId) continue;
      return item;
    }
    return null;
  }

  // Starts the synthesis request immediately and returns a live stream handle. Audio blobs are
  // handed to the player as soon as their NDJSON line is parsed, so playback of the first chunk
  // overlaps synthesis of the rest instead of waiting for the reader loop to end.
  function prepareTtsStreamItem(item) {
    ensureTtsBargeInListening(item);
    const stream = {
      item,
      audioItems: [],
      cursor: 0,
      finished: false,
      cancelled: false,
      controller: new AbortController(),
      waiters: [],
    };
    // Nothing awaits the reader loop: the player consumes the stream instead, so the promise
    // needs its own catch to stay off the unhandled-rejection path.
    stream.completion = runTtsStreamItem(stream).catch(() => finishTtsStream(stream));
    return stream;
  }

  async function runTtsStreamItem(stream) {
    const { item, controller } = stream;
    const timeoutMs = ttsSynthesisTimeoutMs(item.text, item.voiceId);
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
    ttsCurrentAbortControllersRef.current.add(controller);
    recordVoiceMetric(item.token, 'firstFetchAt', Date.now(), { sessionId: item.sessionId });
    try {
      const response = await fetch('/api/tts/stream', {
        method: 'POST',
        headers: { ...API_HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: item.text,
          voiceId: item.voiceId || 'default',
          sessionId: item.sessionId,
          turnId: item.token,
          segmentIndex: item.segmentIndex ?? 0,
        }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const detail = await response.text().catch(() => '');
        if (activeSessionRef.current === item.sessionId) {
          setVoiceError(format(t.voiceUnavailable, { error: detail || `${response.status} ${response.statusText}` }));
        }
        return;
      }
      if (!response.body) {
        pushTtsStreamAudio(stream, await response.blob(), {});
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (consumeTtsStreamLine(stream, line)) return;
        }
      }
      if (buffer.trim()) consumeTtsStreamLine(stream, buffer);
    } catch (err) {
      if (err.name !== 'AbortError') {
        if (activeSessionRef.current === item.sessionId) {
          setVoiceError(format(t.voiceUnavailable, { error: err.message || t.requestFailed }));
        }
      }
    } finally {
      window.clearTimeout(timeoutId);
      ttsCurrentAbortControllersRef.current.delete(controller);
      finishTtsStream(stream);
    }
  }

  // Returns true when the stream is terminated by the event (a `cancelled` notice ends the
  // response, and it must not raise an error toast).
  function consumeTtsStreamLine(stream, line) {
    const payload = decodeTtsStreamLine(line, stream.item);
    if (!payload) return false;
    if (payload.type === 'cancelled') {
      stream.cancelled = true;
      return true;
    }
    if (payload.type !== 'audio' || !payload.audio) return false;
    if (!claimTtsStreamAudioKey(stream.item, payload.seq)) return false;
    pushTtsStreamAudio(stream, base64AudioBlob(payload.audio, payload.contentType || 'audio/wav'), payload);
    return false;
  }

  function pushTtsStreamAudio(stream, blob, payload) {
    if (!blob) return;
    if (stream.audioItems.length === 0) {
      recordVoiceMetric(stream.item.token, 'firstAudioAt', Date.now(), {
        sessionId: stream.item.sessionId,
        firstAudioMs: Number(payload?.firstAudioMs) || 0,
      });
    }
    stream.audioItems.push(createPreparedTtsAudio(blob));
    wakeTtsStreamWaiters(stream);
  }

  function finishTtsStream(stream) {
    if (stream.finished) return;
    stream.finished = true;
    wakeTtsStreamWaiters(stream);
  }

  function wakeTtsStreamWaiters(stream) {
    const waiters = stream.waiters;
    stream.waiters = [];
    for (const resolve of waiters) resolve();
  }

  // Resolves once the first audio blob has landed, or the stream ended without one. The pump
  // uses it as the "prepared" checkpoint, so a segment is considered ready at first audio.
  async function waitForFirstTtsStreamAudio(stream) {
    while (stream.audioItems.length === 0 && !stream.finished) {
      await new Promise((resolve) => stream.waiters.push(resolve));
    }
    return stream;
  }

  async function takeNextTtsStreamAudio(stream) {
    while (stream.cursor >= stream.audioItems.length) {
      if (stream.finished) return null;
      await new Promise((resolve) => stream.waiters.push(resolve));
    }
    const audioItem = stream.audioItems[stream.cursor];
    stream.cursor += 1;
    return audioItem;
  }

  // Dedupes played audio by (turnId, segmentIndex, seq) so a replayed or retried NDJSON event
  // can never speak the same fragment twice.
  function claimTtsStreamAudioKey(item, seq) {
    if (seq === undefined || seq === null) return true;
    const key = `${item.token}|${item.segmentIndex ?? 0}|${seq}`;
    const played = ttsPlayedAudioKeysRef.current;
    if (played.has(key)) return false;
    if (played.size >= TTS_PLAYED_AUDIO_KEY_LIMIT) played.clear();
    played.add(key);
    return true;
  }

  function createPreparedTtsAudio(blob) {
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.preload = 'auto';
    try {
      audio.load();
    } catch {
      // Loading is advisory; play() will still surface errors if decoding fails.
    }
    return { blob, url, audio };
  }

  function decodeTtsStreamLine(line, item) {
    if (!line.trim() || item.token !== ttsPlaybackTokenRef.current) return null;
    const payload = JSON.parse(line);
    if (payload.type === 'error') {
      if (activeSessionRef.current === item.sessionId) {
        setVoiceError(format(t.voiceUnavailable, { error: payload.error || t.requestFailed }));
      }
      return null;
    }
    return payload;
  }

  async function playPreparedTtsItem(prepared) {
    if (!claimPreparedTtsPlayback(prepared.item)) {
      releasePreparedTts(prepared);
      return;
    }
    if (prepared.item.delayDisplay) {
      appendDelayedAssistantContent(prepared.item.sessionId, prepared.item.assistantId, prepared.item.displayText || prepared.item.text);
    }
    while (isCurrentTtsSpeechItem(prepared.item)) {
      const audioItem = await takeNextTtsStreamAudio(prepared);
      if (!audioItem) break;
      if (!isCurrentTtsSpeechItem(prepared.item)) {
        releasePreparedTtsAudio(audioItem);
        break;
      }
      await playPreparedTtsAudio(audioItem, prepared.item);
    }
    releasePreparedTts(prepared);
  }

  function releaseDelayedDisplayOnTtsFailure(item) {
    if (!item?.delayDisplay) return;
    if (!claimPreparedTtsPlayback(item)) return;
    appendDelayedAssistantContent(item.sessionId, item.assistantId, item.displayText || item.text);
  }

  function claimPreparedTtsPlayback(item) {
    if (!isCurrentTtsSpeechItem(item)) return false;
    const token = item.token;
    const normalized = item.normalizedText || normalizeTtsSpeechForDedupe(item.text);
    const state = ttsPlayedSpeechRef.current[token] || { ids: new Set(), spoken: '' };
    if (item.speechId && state.ids.has(item.speechId)) return false;
    if (normalized && state.spoken && state.spoken.includes(normalized)) return false;
    if (item.speechId) state.ids.add(item.speechId);
    state.spoken = appendNormalizedTtsSpeech(state.spoken || '', normalized);
    ttsPlayedSpeechRef.current[token] = state;
    return true;
  }

  async function playPreparedTtsAudio(audioItem, item) {
    if (
      item.token !== ttsPlaybackTokenRef.current ||
      ttsSuppressedTokensRef.current.has(item.token) ||
      activeSessionRef.current !== item.sessionId
    ) {
      releasePreparedTtsAudio(audioItem);
      return;
    }
    ensureTtsBargeInListening(item);
    recordTtsPlaybackStartMetrics(item.token, item.sessionId);
    await new Promise((resolve) => {
      const { audio, url } = audioItem;
      ttsObjectUrlRef.current = url;
      ttsAudioRef.current = audio;
      ttsPlayingRef.current = true;
      // A duck decided between two chunks must carry over to the element that starts next.
      audio.volume = ttsDuckedRef.current ? TTS_DUCK_VOLUME : 1;
      const finish = () => {
        if (ttsAudioResolveRef.current === finish) ttsAudioResolveRef.current = null;
        ttsPlaybackEndedAtRef.current = Date.now();
        if (ttsAudioRef.current === audio) {
          ttsPlayingRef.current = false;
          stopCurrentTtsAudio();
        } else {
          URL.revokeObjectURL(url);
        }
        resolve();
      };
      ttsAudioResolveRef.current = finish;
      audio.onended = finish;
      audio.onerror = () => {
        if (activeSessionRef.current === item.sessionId) {
          setVoiceError(format(t.voiceUnavailable, { error: t.requestFailed }));
        }
        finish();
      };
      audio.play().catch((err) => {
        if (activeSessionRef.current === item.sessionId) {
          setVoiceError(format(t.voiceUnavailable, { error: err.message || t.requestFailed }));
        }
        finish();
      });
    });
  }

  function releasePreparedTts(prepared) {
    if (!prepared) return;
    try {
      prepared.controller?.abort();
    } catch {
      // The controller may already be closed by the fetch path.
    }
    // Everything before the cursor already played and released itself.
    for (let index = prepared.cursor || 0; index < (prepared.audioItems?.length || 0); index += 1) {
      releasePreparedTtsAudio(prepared.audioItems[index]);
    }
    prepared.cursor = prepared.audioItems?.length || 0;
  }

  function releasePreparedTtsAudio(audioItem) {
    if (!audioItem?.url) return;
    try {
      audioItem.audio?.pause();
    } catch {
      // Best effort cleanup.
    }
    URL.revokeObjectURL(audioItem.url);
    audioItem.url = '';
  }

  function suppressTtsToken(token) {
    if (!token) return;
    ttsSuppressedTokensRef.current.add(token);
    requestTtsTurnCancel(token);
    ttsQueueRef.current = ttsQueueRef.current.filter((item) => item.token !== token);
    delete ttsChunkBuffersRef.current[token];
    delete ttsIncomingTextRef.current[token];
    delete ttsQueuedSpeechRef.current[token];
    delete ttsPlayedSpeechRef.current[token];
    delete ttsSpeechSequenceRef.current[token];
    delete ttsDelayedDisplayRef.current[token];
    forgetTtsPlayedAudioKeys(token);
    if (ttsPlaybackTokenRef.current === token) {
      ttsPlaybackTokenRef.current = '';
      ttsBargeInTokenRef.current = '';
    }
  }

  // Tells the backend to stop synthesising that turn; aborting the fetches alone leaves the
  // producer thread running until it finishes the whole segment.
  function requestTtsTurnCancel(token) {
    const parsed = parseTtsToken(token);
    if (!parsed?.sessionId) return;
    void api('/api/tts/cancel', { sessionId: parsed.sessionId, turnId: token }).catch(() => {});
  }

  function forgetTtsPlayedAudioKeys(token) {
    if (!token) return;
    const prefix = `${token}|`;
    for (const key of ttsPlayedAudioKeysRef.current) {
      if (key.startsWith(prefix)) ttsPlayedAudioKeysRef.current.delete(key);
    }
  }

  // Stage one of barge-in: the backend heard speech, so drop the volume without giving up the
  // playback position. Stage two (a confirmed intent) goes through stopTtsPlayback instead.
  function duckTtsPlayback() {
    if (ttsDuckedRef.current) return;
    // Speech can start in the gap between two chunks; the flag makes playPreparedTtsAudio
    // bring the next element in already ducked.
    if (!ttsPlayingRef.current && !ttsPlaybackTokenRef.current) return;
    ttsDuckedRef.current = true;
    const audio = ttsAudioRef.current;
    if (!audio) return;
    // Kept so the metrics line can show where the duck landed; playback is never rewound.
    ttsDuckedPositionRef.current = audio.currentTime || 0;
    recordVoiceMetric(currentTtsToken(), 'duckedAtSec', ttsDuckedPositionRef.current || -1);
    audio.volume = TTS_DUCK_VOLUME;
  }

  function restoreTtsPlaybackVolume() {
    if (!ttsDuckedRef.current) return;
    ttsDuckedRef.current = false;
    ttsDuckedPositionRef.current = 0;
    const audio = ttsAudioRef.current;
    if (audio) audio.volume = 1;
  }

  function isBackendSpeechActive() {
    if (!voiceSpeechActiveRef.current) return false;
    const startedAt = voiceSpeechStartedAtRef.current;
    return Boolean(startedAt) && Date.now() - startedAt < VOICE_SPEECH_HOLD_MAX_MS;
  }

  // Stage two of barge-in. An explicit interrupt phrase always counts; substantial speech only
  // counts inside the confirmation window that opened when the backend VAD reported speech.
  function shouldTriggerBargeIn(partialText) {
    if (isBargeInIntent(partialText)) return true;
    const startedAt = voiceSpeechStartedAtRef.current;
    if (!startedAt || Date.now() - startedAt > BARGE_IN_CONFIRM_WINDOW_MS) return false;
    return isSubstantialBargeInSpeech(partialText);
  }

  function ttsSynthesisTimeoutMs(text, voiceId = 'default') {
    const length = String(text || '').length;
    if (voiceId && voiceId !== 'default') {
      return Math.max(45000, Math.min(180000, 30000 + length * 1800));
    }
    return Math.max(9000, Math.min(18000, 7000 + length * 140));
  }

  async function playVoicePreview() {
    const voice = ttsVoiceOptions.find((item) => item.id === ttsVoiceId);
    if (!voice?.previewUrl) return;
    setVoiceError('');
    try {
      if (ttsPreviewAudioRef.current) {
        ttsPreviewAudioRef.current.pause();
        ttsPreviewAudioRef.current.currentTime = 0;
      }
      const response = await fetch(`${voice.previewUrl}?t=${Date.now()}`, { headers: API_HEADERS });
      if (!response.ok) {
        const detail = await response.text().catch(() => '');
        throw new Error(detail || `${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      ttsPreviewAudioRef.current = audio;
      audio.addEventListener(
        'ended',
        () => {
          URL.revokeObjectURL(url);
          if (ttsPreviewAudioRef.current === audio) ttsPreviewAudioRef.current = null;
        },
        { once: true },
      );
      audio.addEventListener(
        'error',
        () => {
          URL.revokeObjectURL(url);
          if (ttsPreviewAudioRef.current === audio) ttsPreviewAudioRef.current = null;
        },
        { once: true },
      );
      await audio.play();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setVoiceError(format(t.voicePreviewFailed, { error: message || t.requestFailed }));
    }
  }

  async function approve(type, options = {}) {
    const approvalSessionId = sessionId;
    if (!pendingApproval || busy || runningSessions[approvalSessionId]) return;
    const approvalAssistantId = pendingApprovalAssistantId || findLastAssistantId(messages);
    const runId = crypto.randomUUID();
    markSessionRunning(approvalSessionId, true);
    setError('');
    let approval = { type };
    if (options.remember) approval.remember = true;
    if (type === 'reject') approval.reason = approvalEdit || t.rejectedFromUi;
    if (type === 'feedback') approval.feedback = approvalEdit || t.reviseTryAgain;
    if (type === 'edit') {
      try {
        approval.tool_input = JSON.parse(approvalEdit || '{}');
      } catch (err) {
        markSessionRunning(approvalSessionId, false);
        setError(format(t.invalidApprovalJson, { error: err.message }));
        return;
      }
    }
    const processingLabel = approvalProcessingText(pendingApproval, t, type);
    const makeRunningProgress = (current) => ({
      ...current,
      current: { status: 'running', label: processingLabel },
    });
    const runningMessages = approvalAssistantId
      ? setAssistantProgressRunningInMessages(
          updateAssistantProgressInMessages(messages, approvalAssistantId, makeRunningProgress),
          approvalAssistantId,
          true,
        )
      : messages;
    activeRunsRef.current[approvalSessionId] = {
      runId,
      assistantId: approvalAssistantId,
      controller: null,
      initialMessages: runningMessages,
    };
    sessionLiveMessagesRef.current[approvalSessionId] = runningMessages;
    if (approvalAssistantId) {
      setMessages(runningMessages);
    } else {
      updateLatestAssistantProgress(makeRunningProgress);
    }
    delete sessionPendingApprovalsRef.current[approvalSessionId];
    setPendingApproval(null);
    setPendingApprovalAssistantId('');
    setApprovalEdit('');
    setApprovalProcessing(true);
    try {
      await streamChat({ sessionId: approvalSessionId, approval }, approvalAssistantId, approvalSessionId, runId, '/api/approval-stream');
      if (activeRunsRef.current[approvalSessionId]?.runId === runId) {
        updateSessionMessages(approvalSessionId, (current) =>
          setAssistantProgressRunningInMessages(
            updateAssistantProgressInMessages(current, approvalAssistantId, (progress) => finishAgentProgress(progress, t)),
            approvalAssistantId,
            false,
          ),
        );
        clearRunIfCurrent(approvalSessionId, runId);
      }
    } catch (err) {
      if (activeSessionRef.current === approvalSessionId) {
        setError(format(t.streamFailed, { error: err.message }));
        if (approvalAssistantId) {
          updateAssistantProgress(approvalAssistantId, (current) => ({ ...current, current: null }));
          setAssistantProgressRunning(approvalAssistantId, false);
        } else {
          updateLatestAssistantProgress((current) => ({ ...current, current: null }));
        }
      }
    } finally {
      if (activeSessionRef.current === approvalSessionId) {
        setApprovalProcessing(false);
      }
      refreshSessions();
    }
  }

  async function chooseWorkspaceWithNativePicker() {
    setError('');
    const nativeResponse = await api('/api/directory/native', {
      start: selectedWorkspace || workspaceInput || status?.workspace,
      prompt: t.nativePickerPrompt,
    });
    if (nativeResponse.ok && nativeResponse.cancelled) return;
    if (nativeResponse.ok && nativeResponse.path) {
      return nativeResponse.path;
    }

    setError(nativeResponse.error || t.requestFailed);
    return null;
  }

  async function createSessionForWorkspace(workspacePath) {
    const next = `web-${Date.now().toString(36)}`;
    const response = await api('/api/session/create', { sessionId: next, workspace: workspacePath });
    if (!response.ok) {
      setError(response.error || t.requestFailed);
      return;
    }
    setSessions(response.sessions || []);
    setCollapsedWorkspaces((current) => ({ ...current, [workspacePath]: false }));
    await openSession(next);
  }

  async function saveSettings(event) {
    event.preventDefault();
    const selected = findModelOption(modelInput) || makeModelOption(status?.provider, status?.model);
    if (!selected.model.trim() || workspaceBusy) return;
    setBusy(true);
    setError('');
    const response = await api('/api/settings', {
      provider: selected.provider,
      model: selected.model,
      gateway: selected.gateway || '',
      thinking: Boolean(selected.supportsThinking && thinkingEnabled),
    });
    setBusy(false);
    if (!response.ok) {
      setError(response.error || t.requestFailed);
      return;
    }
    setStatus(response);
    setThinkingEnabled(Boolean(response.thinking));
    setModelInput(
      modelOptionValue(
        response.provider || selected.provider,
        response.model || selected.model,
        response.gateway || selected.gateway || '',
      ),
    );
    setSettingsOpen(false);
  }

  function applyResponse(response, targetAssistantId = '') {
    if (!response.ok) {
      setError(response.error || t.requestFailed);
    }
    if (response.messages?.length) {
      setMessages((current) => appendResponseMessages(current, response.messages, targetAssistantId));
    }
    if (response.toolResult) {
      const toolContent = JSON.stringify(response.toolResult, null, 2);
      if (!shouldHideToolResult(toolContent)) {
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            role: 'tool',
            kind: 'tool_result',
            content: toolContent,
          },
        ]);
      }
    }
    if (response.pendingApproval) {
      setPendingApproval(response.pendingApproval);
      setPendingApprovalAssistantId(targetAssistantId || findLastAssistantId(messages));
      setApprovalEdit(JSON.stringify(response.pendingApproval.toolInput, null, 2));
    }
    if (Array.isArray(response.todos)) {
      if (targetAssistantId) {
        setAssistantTodos(targetAssistantId, response.todos);
      } else {
        updateLatestAssistantTodos(response.todos);
      }
    }
  }

  async function streamChat(payload, assistantId, streamSessionId, runId, endpoint = '/api/chat-stream') {
    const controller = new AbortController();
    if (activeRunsRef.current[streamSessionId]?.runId !== runId) return;
    activeRunsRef.current[streamSessionId].controller = controller;
    streamControllersRef.current[streamSessionId] = controller;
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { ...API_HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, runId }),
        signal: controller.signal,
      });
      // The server may have restarted since this page was served; a 403 here
      // means a stale token, and reloading is what fixes it.
      if (handleUnauthorizedResponse(response)) return;
      if (!response.body) {
        setError(t.streamUnavailable);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.trim()) handleStreamEvent(JSON.parse(line), assistantId, streamSessionId, runId);
        }
      }
      if (buffer.trim()) handleStreamEvent(JSON.parse(buffer), assistantId, streamSessionId, runId);
    } catch (err) {
      if (err.name === 'AbortError') {
        return;
      }
      const responseAssistantId = activeRunsRef.current[streamSessionId]?.responseAssistantId || assistantId;
      if (activeSessionRef.current === streamSessionId) {
        setError(format(t.streamFailed, { error: err.message }));
        setAssistantProgressRunning(assistantId, false);
      }
      if (shouldDelayCustomVoiceDisplay(streamSessionId, responseAssistantId)) {
        queueAssistantTtsDelta(streamSessionId, responseAssistantId, '', {
          force: true,
          delayDisplay: true,
        });
        setAssistantVoiceTtsPreparing(streamSessionId, responseAssistantId, false);
      }
    } finally {
      if (activeRunsRef.current[streamSessionId]?.runId === runId) {
        activeRunsRef.current[streamSessionId].controller = null;
      }
    }
  }

  function handleStreamEvent(event, assistantId, streamSessionId, runId) {
    if (activeRunsRef.current[streamSessionId]?.runId !== runId) return;
    const isActiveSession = activeSessionRef.current === streamSessionId;
    const responseAssistantId = activeRunsRef.current[streamSessionId]?.responseAssistantId || assistantId;
    if (event.type === 'progress') {
      updateSessionMessages(streamSessionId, (current) =>
        updateAssistantProgressInMessages(current, assistantId, (progress) => updateAgentProgress(progress, event, t)),
      );
      return;
    }
    if (event.type === 'heartbeat') {
      const waited = Number(event.waitedSec);
      updateSessionMessages(streamSessionId, (current) =>
        setAssistantHeartbeatInMessages(current, responseAssistantId, Number.isFinite(waited) ? waited : 0),
      );
      return;
    }
    if (event.type === 'usage') {
      updateSessionMessages(streamSessionId, (current) =>
        setAssistantUsageInMessages(current, responseAssistantId, {
          inputTokens: Number(event.inputTokens) || 0,
          outputTokens: Number(event.outputTokens) || 0,
          totalTokens: Number(event.totalTokens) || 0,
        }),
      );
      return;
    }
    if (event.type === 'notice') {
      const noticeText = String(event.message || '').trim();
      if (noticeText && isActiveSession) pushToast({ message: noticeText, tone: 'info' });
      if (event.kind === 'context_compacted') {
        updateSessionMessages(streamSessionId, (current) => [
          ...current,
          { id: crypto.randomUUID(), role: 'system', kind: 'divider', content: t.contextCompacted },
        ]);
      }
      return;
    }
    if (event.type === 'delta') {
      updateSessionMessages(streamSessionId, (current) =>
        setAssistantHeartbeatInMessages(current, responseAssistantId, null),
      );
      if (shouldDelayCustomVoiceDisplay(streamSessionId, responseAssistantId)) {
        setAssistantVoiceTtsPreparing(streamSessionId, responseAssistantId, true);
        queueAssistantTtsDelta(streamSessionId, responseAssistantId, event.content || '', { delayDisplay: true });
        return;
      }
      queueDisplayDelta(streamSessionId, responseAssistantId, event.content || '');
      queueAssistantTtsDelta(streamSessionId, responseAssistantId, event.content || '');
      return;
    }
    if (event.type === 'thinking_delta') {
      updateSessionMessages(streamSessionId, (current) =>
        appendAssistantThinkingInMessages(current, assistantId, event.content || ''),
      );
      return;
    }
    if (event.type === 'tool_result') {
      if (event.kind === 'diagram') {
        updateSessionMessages(streamSessionId, (current) =>
          ensurePostToolAssistantMessage(
            [
              ...current,
              {
                id: crypto.randomUUID(),
                role: 'assistant',
                kind: 'diagram',
                title: event.title || '',
                diagramType: event.diagramType || 'flowchart',
                content: event.content || '',
              },
            ],
            activeRunsRef.current[streamSessionId],
          ),
        );
        return;
      }
      if (event.kind === 'agent_dialogue') {
        updateSessionMessages(streamSessionId, (current) =>
          ensurePostToolAssistantMessage(upsertAgentDialogueMessage(current, event), activeRunsRef.current[streamSessionId]),
        );
        return;
      }
      if (event.ok === true) {
        const preview = String(event.preview || '');
        if (preview) {
          updateSessionMessages(streamSessionId, (current) =>
            updateAssistantProgressInMessages(current, assistantId, (progress) =>
              attachToolPreviewToProgress(progress, event),
            ),
          );
        }
        return;
      }
      if (shouldHideToolResult(event.content)) return;
      updateSessionMessages(streamSessionId, (current) =>
        ensurePostToolAssistantMessage(
          [
            ...current,
            {
              id: crypto.randomUUID(),
              role: 'tool',
              kind: 'tool_result',
              content: event.content,
            },
          ],
          activeRunsRef.current[streamSessionId],
        ),
      );
      return;
    }
    if (event.type === 'todos') {
      const nextTodos = Array.isArray(event.todos) ? event.todos : [];
      updateSessionMessages(streamSessionId, (current) =>
        updateAssistantProgressInMessages(
          setAssistantTodosInMessages(current, assistantId, nextTodos),
          assistantId,
          (progress) => ({ ...progress, summary: event.summary || progress.summary }),
        ),
      );
      return;
    }
    if (event.type === 'pending_approval') {
      sessionPendingApprovalsRef.current[streamSessionId] = event.pendingApproval;
      flushQueuedDisplayDelta(streamSessionId, responseAssistantId, { flushMarkdown: true });
      stopTtsPlayback({
        clearQueue: true,
        stopVoice: true,
        suppressCurrent: true,
        suppressToken: ttsToken(streamSessionId, responseAssistantId),
      });
      if (voiceActiveRef.current) void stopVoiceCapture({ cancel: true, submit: false });
      if (isActiveSession) {
        setPendingApproval(event.pendingApproval);
        setPendingApprovalAssistantId(assistantId);
        setApprovalEdit(JSON.stringify(event.pendingApproval.toolInput, null, 2));
      }
      return;
    }
    if (event.type === 'done') {
      if (!event.cancelled) {
        delete sessionPendingApprovalsRef.current[streamSessionId];
      }
      const delayCustomVoiceDisplay =
        !event.cancelled && isActiveSession && shouldDelayCustomVoiceDisplay(streamSessionId, responseAssistantId);
      if (!event.cancelled && !delayCustomVoiceDisplay) {
        flushQueuedDisplayDelta(streamSessionId, responseAssistantId, { flushMarkdown: true });
      }
      updateSessionMessages(streamSessionId, (current) =>
        setAssistantProgressRunningInMessages(
          updateAssistantProgressInMessages(current, assistantId, (progress) => finishAgentProgress(progress, t)),
          assistantId,
          false,
        ).map((message) =>
          message.id === responseAssistantId
            ? {
                ...message,
                thinkingRunning: false,
                voiceTtsPreparing: delayCustomVoiceDisplay ? message.voiceTtsPreparing : false,
              }
            : message,
        ),
      );
      if (!event.cancelled && isActiveSession) {
        queueAssistantTtsDelta(streamSessionId, responseAssistantId, '', {
          force: true,
          delayDisplay: delayCustomVoiceDisplay,
        });
      }
      if (!event.cancelled && isActiveSession && voiceConversationActiveRef.current && !ttsEnabledRef.current) {
        if (voiceActiveRef.current) void stopVoiceCapture({ cancel: true, submit: false });
        scheduleVoiceConversationListen(350);
      }
      if (event.cancelled) {
        clearMarkdownDeltaBuffer(streamSessionId, responseAssistantId);
        setAssistantVoiceTtsPreparing(streamSessionId, responseAssistantId, false);
        stopTtsPlayback({
          clearQueue: true,
          stopVoice: true,
          suppressCurrent: true,
          suppressToken: ttsToken(streamSessionId, responseAssistantId),
        });
      }
      clearRunIfCurrent(streamSessionId, runId);
      if (!isActiveSession) {
        markSessionUnread(streamSessionId);
        refreshSessions();
      }
      return;
    }
    if (event.type === 'error') {
      flushQueuedDisplayDelta(streamSessionId, responseAssistantId, { flushMarkdown: true });
      stopVoiceConversation();
      if (isActiveSession) {
        const hint = errorCodeHint(event.code, t);
        const baseMessage = String(event.error || event.message || '').trim() || t.requestFailed;
        const message = hint ? `${baseMessage} · ${hint}` : baseMessage;
        const failedRun = activeRunsRef.current[streamSessionId];
        const retryText = failedRun?.text || lastUserTextRef.current;
        // The retry has to be pinned to the session and the user message of THIS run: the
        // rendered session may have changed by the time the button is pressed.
        pushToast({
          message,
          tone: 'error',
          fatal: true,
          retry:
            event.retriable && retryText
              ? {
                  sessionId: streamSessionId,
                  userMessageId: failedRun?.userMessageId || '',
                  assistantId: responseAssistantId,
                  text: retryText,
                }
              : null,
        });
      }
      updateSessionMessages(streamSessionId, (current) =>
        setAssistantProgressRunningInMessages(
          updateAssistantProgressInMessages(current, assistantId, (progress) => ({ ...progress, current: null })),
          assistantId,
          false,
        ).map((message) =>
          message.id === responseAssistantId ? { ...message, thinkingRunning: false, voiceTtsPreparing: false } : message,
        ),
      );
      clearRunIfCurrent(streamSessionId, runId);
      delete sessionPendingApprovalsRef.current[streamSessionId];
      if (!isActiveSession) {
        markSessionUnread(streamSessionId);
        refreshSessions();
      }
    }
    // Unknown event types are ignored on purpose so a newer backend never breaks the UI.
  }

  async function retryFailedTurn(toast) {
    const retry = toast?.retry;
    dismissToast(toast?.id);
    if (!retry?.text) return;
    const retrySessionId = retry.sessionId || activeSessionRef.current;
    if (!retrySessionId) return;
    if (activeSessionRef.current !== retrySessionId) {
      await openSession(retrySessionId);
      if (activeSessionRef.current !== retrySessionId) return;
    }
    // The backend already dropped the failed turn, so resending the text is right; the UI
    // must reuse the existing user bubble instead of adding a second one, and the failed
    // turn's empty assistant bubble has to go.
    await submitMessage(retry.text, {
      clearInput: false,
      targetSessionId: retrySessionId,
      reuseUserMessageId: retry.userMessageId,
      dropAssistantMessageId: retry.assistantId,
      // activeRunsRef is the authoritative "already running" check and is cleared on error;
      // the runningSessions map can still hold a stale entry for the session we switched to.
      ignoreStaleRunning: true,
    });
  }

  function markSessionUnread(targetSessionId) {
    if (!targetSessionId || targetSessionId === activeSessionRef.current) return;
    setUnreadSessions((current) => (current[targetSessionId] ? current : { ...current, [targetSessionId]: true }));
  }

  function voiceWsUrl() {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${scheme}//${window.location.host}/api/asr/stream?token=${encodeURIComponent(API_TOKEN)}`;
  }

  function clearVoiceRestartTimer() {
    if (!voiceRestartTimerRef.current) return;
    window.clearTimeout(voiceRestartTimerRef.current);
    voiceRestartTimerRef.current = null;
  }

  function scheduleVoiceConversationListen(delay = 350) {
    clearVoiceRestartTimer();
    if (!voiceConversationActiveRef.current || voiceActiveRef.current || pendingApprovalRef.current) return;
    voiceRestartTimerRef.current = window.setTimeout(() => {
      voiceRestartTimerRef.current = null;
      if (!voiceConversationActiveRef.current || voiceActiveRef.current || pendingApprovalRef.current) return;
      const activeRun = activeRunsRef.current[activeSessionRef.current];
      if (activeRun || runningSessionsRef.current[activeSessionRef.current]) {
        void startVoiceCapture({ bargeIn: true, context: currentBargeInContext() });
      } else {
        void startVoiceCapture();
      }
    }, delay);
  }

  function startVoiceConversation() {
    if (!sessionId) {
      setError(t.chooseWorkspaceFirst);
      return;
    }
    voiceConversationActiveRef.current = true;
    setVoiceConversationActive(true);
    if (activeSessionBusy || activeRunsRef.current[sessionId]) {
      void startVoiceCapture({ bargeIn: true, forceBargeIn: true, context: currentBargeInContext() });
    } else {
      void startVoiceCapture();
    }
  }

  function stopVoiceConversation() {
    voiceConversationActiveRef.current = false;
    setVoiceConversationActive(false);
    clearVoiceRestartTimer();
    const activeRun = activeRunsRef.current[activeSessionRef.current];
    stopTtsPlayback({
      clearQueue: true,
      stopVoice: false,
      suppressCurrent: true,
      suppressToken: activeRun?.assistantId ? ttsToken(activeSessionRef.current, activeRun.assistantId) : '',
    });
    void stopVoiceCapture({ cancel: true, submit: false });
  }

  async function startVoiceCapture(options = {}) {
    const { bargeIn = false, context = null, forceBargeIn = false } = options;
    if (!sessionId) {
      setError(t.chooseWorkspaceFirst);
      return;
    }
    if (voiceActiveRef.current || pendingApprovalRef.current || (activeSessionBusy && !bargeIn)) return;
    setVoiceError('');
    setError('');
    setVoiceStatus(bargeIn ? t.bargeInListening : t.voiceStarting);
    setVoiceActive(true);
    voiceActiveRef.current = true;
    voiceStoppingRef.current = false;
    voiceSessionIdRef.current = sessionId;
    voiceDraftUserIdRef.current = '';
    voiceLastPartialTextRef.current = '';
    voiceLastPartialChangedAtRef.current = 0;
    voiceSpeechActiveRef.current = false;
    voiceSpeechStartedAtRef.current = 0;
    clearVoiceAutoFinishTimer();
    bargeInTriggeredRef.current = Boolean(bargeIn && forceBargeIn);
    bargeInContextRef.current = bargeIn ? context : null;
    voiceBufferRef.current = new Float32Array(0);

    try {
      const ws = new WebSocket(voiceWsUrl());
      ws.binaryType = 'arraybuffer';
      voiceSocketRef.current = ws;
      await waitForSocketOpen(ws);
      ws.send(JSON.stringify({ type: 'start', model: voiceModel }));

      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (voiceSocketRef.current !== ws) return;
        if (payload.type === 'ready') {
          setVoiceStatus(t.voiceListening);
          return;
        }
        // Stage one of barge-in: the backend VAD reports speech before any transcript exists,
        // so duck the assistant instead of cutting it off on a cough or a filler word.
        if (payload.type === 'speech') {
          const speaking = payload.state === 'start';
          voiceSpeechActiveRef.current = speaking;
          if (speaking) {
            voiceSpeechStartedAtRef.current = Date.now();
            if (bargeIn && !bargeInTriggeredRef.current) duckTtsPlayback();
          } else {
            voiceSpeechStartedAtRef.current = 0;
            if (!bargeInTriggeredRef.current) restoreTtsPlaybackVolume();
          }
          return;
        }
        if (payload.type === 'partial') {
          const partialText = String(payload.text || '').trim();
          setVoiceStatus(partialText ? t.voiceTranscribing : bargeIn ? t.bargeInListening : t.voiceListening);
          if (partialText) {
            if (voiceLastPartialTextRef.current !== partialText) {
              voiceLastPartialTextRef.current = partialText;
              voiceLastPartialChangedAtRef.current = Date.now();
            } else if (!voiceLastPartialChangedAtRef.current) {
              voiceLastPartialChangedAtRef.current = Date.now();
            }
            if (bargeIn && !bargeInTriggeredRef.current && shouldTriggerBargeIn(partialText)) {
              bargeInTriggeredRef.current = true;
              recordVoiceMetric(currentTtsToken(), 'bargeInDetectedAt', Date.now());
              setVoiceStatus(t.bargeInDetected);
              stopTtsPlayback({ clearQueue: true, stopVoice: false, suppressCurrent: true });
              if (activeRunsRef.current[voiceSessionIdRef.current]) stopGeneration({ stopVoice: false });
            }
            if (!bargeIn || bargeInTriggeredRef.current) {
              updateComposerInput(partialText);
              upsertVoiceDraftMessage(partialText);
              scheduleVoiceAutoFinish(payload, { bargeIn });
            } else {
              clearVoiceAutoFinishTimer();
            }
          } else {
            clearVoiceAutoFinishTimer();
          }
          return;
        }
        if (payload.type === 'final') {
          const finalText = String(payload.text || voiceLastPartialTextRef.current).trim();
          const semanticState = payload.semanticVad?.state || '';
          clearVoiceFinishFallbackTimer();
          clearVoiceAutoFinishTimer();
          voiceSocketRef.current = null;
          try {
            ws.close();
          } catch {
            // Already closed by the browser or the server.
          }
          submitVoiceCaptureText(finalText, { bargeIn, semanticState });
          return;
        }
        if (payload.type === 'error') {
          const message = payload.error || t.requestFailed;
          pushToast({ message: format(t.voiceUnavailable, { error: message }), tone: 'error' });
          void stopVoiceCapture({ cancel: true, submit: false });
        }
      };

      ws.onerror = () => {
        pushToast({ message: format(t.voiceUnavailable, { error: t.requestFailed }), tone: 'error' });
        void stopVoiceCapture({ cancel: true, submit: false });
      };

      await startVoiceAudioPipeline(ws);
    } catch (err) {
      pushToast({
        message: err.name === 'NotAllowedError' ? t.voicePermissionDenied : format(t.voiceUnavailable, { error: err.message }),
        tone: 'error',
      });
      await stopVoiceCapture({ cancel: true, submit: false });
    }
  }

  async function stopVoiceCapture({ cancel = false, submit = true } = {}) {
    const ws = voiceSocketRef.current;
    voiceStoppingRef.current = true;
    clearVoiceAutoFinishTimer();
    setVoiceStatus(submit ? t.voiceFinalizing : '');
    if (cancel || !submit) {
      resetVoiceCaptureState({ clearDraft: cancel });
    }
    await stopVoiceAudioOnly();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: cancel ? 'cancel' : 'finish' }));
      if (cancel) ws.close();
    } else if (ws && cancel) {
      try {
        ws.close();
      } catch {
        // The socket may already be closed by the browser or server.
      }
    }
    if (cancel) voiceSocketRef.current = null;
  }

  function resetVoiceCaptureState({ clearDraft = false } = {}) {
    clearVoiceAutoFinishTimer();
    if (clearDraft) clearVoiceDraftMessage();
    setVoiceActive(false);
    voiceActiveRef.current = false;
    setVoiceStatus('');
    bargeInTriggeredRef.current = false;
    bargeInContextRef.current = null;
    voiceLastPartialTextRef.current = '';
    voiceLastPartialChangedAtRef.current = 0;
    // A capture that ends while ducked must not leave the next reply at 20% volume.
    voiceSpeechActiveRef.current = false;
    voiceSpeechStartedAtRef.current = 0;
    restoreTtsPlaybackVolume();
  }

  function submitVoiceCaptureText(finalText, { bargeIn = false, semanticState = '' } = {}) {
    const text = String(finalText || '').trim();
    if (isDiscardableVoiceFinal(text, semanticState) || (bargeIn && !bargeInTriggeredRef.current)) {
      clearVoiceDraftMessage();
      void stopVoiceAudioOnly();
      setVoiceActive(false);
      voiceActiveRef.current = false;
      setVoiceStatus('');
      scheduleVoiceConversationListen();
      return;
    }
    const voiceInterrupt = bargeIn ? composeBargeInToolInput(text, bargeInContextRef.current) : null;
    const submitText = text;
    const draftUserId = upsertVoiceDraftMessage(submitText);
    setVoiceStatus(t.voiceFinalizing);
    void stopVoiceAudioOnly();
    setVoiceActive(false);
    voiceActiveRef.current = false;
    if (submitText) {
      updateComposerInput(bargeIn ? text : submitText);
      if (bargeIn && activeRunsRef.current[voiceSessionIdRef.current]) {
        stopGeneration({ stopVoice: false });
      }
      void submitMessage(submitText, {
        clearInput: true,
        reuseUserMessageId: draftUserId,
        ignoreStaleRunning: bargeIn,
        voiceInterrupt,
      });
      if (voiceConversationActiveRef.current) scheduleVoiceConversationListen(650);
    } else {
      clearVoiceDraftMessage();
      scheduleVoiceConversationListen();
    }
  }

  function clearVoiceFinishFallbackTimer() {
    if (voiceFinishFallbackTimerRef.current) {
      window.clearTimeout(voiceFinishFallbackTimerRef.current);
    }
    voiceFinishFallbackTimerRef.current = null;
  }

  function clearVoiceAutoFinishTimer({ keepFinishFallback = false } = {}) {
    if (voiceAutoFinishTimerRef.current) {
      window.clearTimeout(voiceAutoFinishTimerRef.current);
    }
    voiceAutoFinishTimerRef.current = null;
    voiceAutoFinishTextRef.current = '';
    voiceAutoFinishModeRef.current = '';
    if (!keepFinishFallback) clearVoiceFinishFallbackTimer();
  }

  function isDiscardableVoiceFinal(text, semanticState) {
    if (semanticState !== 'invalid') return false;
    const compact = String(text || '').replace(/\s+/g, '');
    return !compact || compact.length <= 2;
  }

  function scheduleVoiceAutoFinish(payload, { bargeIn = false } = {}) {
    const text = String(payload?.text || '').trim();
    const semanticState = payload?.semanticVad?.state || '';
    if (!text || isDiscardableVoiceFinal(text, semanticState)) {
      clearVoiceAutoFinishTimer();
      return;
    }
    if (bargeIn && !bargeInTriggeredRef.current) return;
    // Once `finish` has been sent we are waiting for the server `final`; a late partial
    // must not clobber the fallback timer.
    if (voiceStoppingRef.current || voiceFinishFallbackTimerRef.current) return;
    const ws = voiceSocketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const isSemanticComplete = semanticState === 'complete';
    const isSemanticIncomplete = semanticState === 'incomplete';
    const mode = isSemanticComplete ? 'semantic' : isSemanticIncomplete ? 'incomplete-stable-text' : 'stable-text';
    const changedAt = voiceLastPartialChangedAtRef.current || Date.now();
    const stableMs = Math.max(0, Date.now() - changedAt);
    const targetStableMs = isSemanticComplete ? 700 : isSemanticIncomplete ? 5200 : 2400;
    const delay = Math.max(isSemanticIncomplete ? 900 : 450, targetStableMs - stableMs);
    if (voiceAutoFinishTimerRef.current && voiceAutoFinishTextRef.current === text && voiceAutoFinishModeRef.current === mode) {
      return;
    }

    clearVoiceAutoFinishTimer();
    voiceAutoFinishTextRef.current = text;
    voiceAutoFinishModeRef.current = mode;
    const finishWhenStable = () => {
      if (!voiceActiveRef.current || voiceStoppingRef.current || voiceSocketRef.current !== ws) return;
      if (voiceLastPartialTextRef.current !== text) return;
      if (Date.now() - changedAt < targetStableMs - 100) return;
      // The backend VAD still hears the user: finishing now would cut the turn in half and
      // race the backend `final`. Re-check instead of giving the timer up. The age bound stops
      // a `speech start` that never gets its `end` from wedging the turn open forever.
      if (isBackendSpeechActive()) {
        voiceAutoFinishTimerRef.current = window.setTimeout(finishWhenStable, 300);
        return;
      }
      setVoiceStatus(t.voiceFinalizing);
      voiceAutoFinishTimerRef.current = null;
      voiceAutoFinishTextRef.current = '';
      voiceAutoFinishModeRef.current = '';
      voiceStoppingRef.current = true;
      void stopVoiceAudioOnly();
      try {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'finish' }));
      } catch {
        voiceSocketRef.current = null;
        submitVoiceCaptureText(text, { bargeIn, semanticState });
        return;
      }
      // Wait up to 800ms for the server `final`; the branch that handles `final`
      // clears this timer explicitly.
      clearVoiceFinishFallbackTimer();
      voiceFinishFallbackTimerRef.current = window.setTimeout(() => {
        voiceFinishFallbackTimerRef.current = null;
        if (voiceSocketRef.current !== ws) return;
        voiceSocketRef.current = null;
        try {
          ws.close();
        } catch {
          // Already closed by the browser or the server.
        }
        submitVoiceCaptureText(text, { bargeIn, semanticState });
      }, 800);
    };
    voiceAutoFinishTimerRef.current = window.setTimeout(finishWhenStable, delay);
  }

  async function stopVoiceAudioOnly() {
    try {
      if (voiceProcessorRef.current) {
        voiceProcessorRef.current.disconnect();
        voiceProcessorRef.current.onaudioprocess = null;
      }
      if (voiceSourceRef.current) voiceSourceRef.current.disconnect();
      if (voiceAudioContextRef.current) await voiceAudioContextRef.current.close();
      if (voiceStreamRef.current) voiceStreamRef.current.getTracks().forEach((track) => track.stop());
    } catch {
      // Best effort cleanup; the next voice session recreates the pipeline.
    }
    voiceProcessorRef.current = null;
    voiceSourceRef.current = null;
    voiceAudioContextRef.current = null;
    voiceStreamRef.current = null;
  }

  async function startVoiceAudioPipeline(ws) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false, // AGC lifts a far-away voice to near-field level, which is exactly what the VAD near-field gate reads.
      },
      video: false,
    });
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextCtor();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    voiceStreamRef.current = stream;
    voiceAudioContextRef.current = audioContext;
    voiceSourceRef.current = source;
    voiceProcessorRef.current = processor;
    processor.onaudioprocess = (event) => {
      if (voiceStoppingRef.current || ws.readyState !== WebSocket.OPEN) return;
      const inputChannel = event.inputBuffer.getChannelData(0);
      const resampled = resampleLinear(inputChannel, audioContext.sampleRate, 16000);
      const next = concatFloat32(voiceBufferRef.current, resampled);
      // 200 ms at 16 kHz: smaller frames let the backend VAD/partials react sooner.
      const chunkSamples = 3200;
      let offset = 0;
      while (next.length - offset >= chunkSamples) {
        const chunk = next.slice(offset, offset + chunkSamples);
        ws.send(chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));
        offset += chunkSamples;
      }
      voiceBufferRef.current = next.slice(offset);
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
  }

  function waitForSocketOpen(ws) {
    return new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = () => reject(new Error(t.requestFailed));
    });
  }

  function concatFloat32(a, b) {
    const output = new Float32Array(a.length + b.length);
    output.set(a, 0);
    output.set(b, a.length);
    return output;
  }

  function resampleLinear(input, sourceRate, targetRate) {
    if (sourceRate === targetRate) return new Float32Array(input);
    const ratio = targetRate / sourceRate;
    const length = Math.max(0, Math.round(input.length * ratio));
    const output = new Float32Array(length);
    for (let index = 0; index < length; index += 1) {
      const sourceIndex = index / ratio;
      const left = Math.floor(sourceIndex);
      const right = Math.min(left + 1, input.length - 1);
      const weight = sourceIndex - left;
      output[index] = input[left] * (1 - weight) + input[right] * weight;
    }
    return output;
  }

  async function newSession() {
    if (busy || creatingSessionRef.current) return;
    creatingSessionRef.current = true;
    setCreatingSession(true);
    try {
      const workspacePath = await chooseWorkspaceWithNativePicker();
      if (!workspacePath) return;
      await createSessionForWorkspace(workspacePath);
    } finally {
      creatingSessionRef.current = false;
      setCreatingSession(false);
    }
  }

  async function openSession(id) {
    if (id === activeSessionRef.current && activeRunsRef.current[id]) {
      setSessionMenu(null);
      return;
    }
    // Never leave a partially buffered frame behind when the feed switches away.
    flushActiveRunDisplayDelta(activeSessionRef.current);
    setUnreadSessions((current) => {
      if (!current[id]) return current;
      const next = { ...current };
      delete next[id];
      return next;
    });
    const activeRun = activeRunsRef.current[id];
    const cachedMessages = activeRun ? sessionLiveMessagesRef.current[id] || activeRun.initialMessages || [] : [];
    if (activeRun && cachedMessages.length > 0) {
      const sessionMeta = visibleSessions.find((session) => session.id === id);
      const cachedApproval = sessionPendingApprovalsRef.current[id] || null;
      activeSessionRef.current = id;
      setSessionId(id);
      // Kept in step with setMessages so callers that act right after an await (retry) never
      // read the previous session's list.
      messagesRef.current = cachedMessages;
      setMessages(cachedMessages);
      setPendingApproval(cachedApproval);
      setPendingApprovalAssistantId(cachedApproval ? activeRun.assistantId || findLastAssistantId(cachedMessages) : '');
      setApprovalEdit(cachedApproval ? JSON.stringify(cachedApproval.toolInput, null, 2) : '');
      setApprovalProcessing(false);
      setError('');
      setInput('');
      setSlashMenuOpen(false);
      setSessionMenu(null);
      setRenamingSessionId('');
      if (sessionMeta?.workspace) {
        setWorkspaceInput(sessionMeta.workspace);
        setSelectedWorkspace(sessionMeta.workspace);
        setStatus((current) => (current ? { ...current, workspace: sessionMeta.workspace } : current));
        setCollapsedWorkspaces((current) => ({ ...current, [sessionMeta.workspace]: false }));
      }
      return;
    }
    activeSessionRef.current = id;
    setSessionId(id);
    messagesRef.current = [];
    setMessages([]);
    setPendingApproval(null);
    setPendingApprovalAssistantId('');
    setApprovalProcessing(false);
    setError('');
    setInput('');
    setSlashMenuOpen(false);
    setSessionMenu(null);
    setRenamingSessionId('');
    const response = await api(`/api/session?sessionId=${encodeURIComponent(id)}`);
    if (activeSessionRef.current !== id) return;
    if (!response.ok) {
      setError(response.error || t.requestFailed);
      messagesRef.current = [];
      setMessages([]);
      return;
    }
    const liveMessages = activeRun ? sessionLiveMessagesRef.current[id] || activeRun.initialMessages : null;
    const loadedMessages = Array.isArray(liveMessages) && liveMessages.length > 0
      ? liveMessages
      : (response.messages || []).map((message) => ({
          ...message,
          id: crypto.randomUUID(),
        }));
    if (activeRun?.assistantId) {
      const lastAssistantIndex = findLastAssistantIndex(loadedMessages);
      if (lastAssistantIndex >= 0) {
        loadedMessages[lastAssistantIndex] = {
          ...loadedMessages[lastAssistantIndex],
          id: activeRun.assistantId,
          progressRunning: true,
        };
      }
    }
    const loadedTodos = Array.isArray(response.todos) ? response.todos : [];
    const restoredMessages = shouldAttachSessionTodos(loadedMessages, loadedTodos, response.pendingApproval)
      ? attachTodosToLastAssistant(loadedMessages, loadedTodos)
      : loadedMessages;
    messagesRef.current = restoredMessages;
    setMessages(restoredMessages);
    if (response.workspace) {
      setWorkspaceInput(response.workspace);
      setSelectedWorkspace(response.workspace);
      setStatus((current) => (current ? { ...current, workspace: response.workspace } : current));
      setCollapsedWorkspaces((current) => ({ ...current, [response.workspace]: false }));
    }
    if (response.pendingApproval) {
      sessionPendingApprovalsRef.current[id] = response.pendingApproval;
      setPendingApproval(response.pendingApproval);
      setApprovalEdit(JSON.stringify(response.pendingApproval.toolInput, null, 2));
    }
  }

  function toggleWorkspaceGroup(workspace) {
    setCollapsedWorkspaces((current) => ({ ...current, [workspace]: !current[workspace] }));
  }

  async function deleteSession(id) {
    const response = await api('/api/session/delete', { sessionId: id });
    if (!response.ok) {
      setError(response.error || t.requestFailed);
      return;
    }
    const nextDraftSessions = draftSessions.filter((session) => session.id !== id);
    setDraftSessions(nextDraftSessions);
    const remainingSessions = response.sessions || [];
    setSessions(remainingSessions);
    setSessionMenu(null);
    setRenamingSessionId((current) => (current === id ? '' : current));
    if (id === sessionId) {
      const remainingVisible = mergeSessions('', remainingSessions, nextDraftSessions);
      if (remainingVisible.length) {
        await openSession(remainingVisible[0].id);
      } else {
        activeSessionRef.current = '';
        setSessionId('');
        setMessages([]);
        setPendingApproval(null);
        setPendingApprovalAssistantId('');
        setError('');
      }
    }
  }

  async function clearSession(id) {
    const response = await api('/api/session/clear', { sessionId: id });
    if (!response.ok) {
      setError(response.error || t.requestFailed);
      return;
    }
    setSessionMenu(null);
    setRenamingSessionId((current) => (current === id ? '' : current));
    setSessions(response.sessions || []);
    setDraftSessions((current) => current.filter((session) => session.id !== id));
    sessionPendingApprovalsRef.current[id] = null;
    const activeRun = activeRunsRef.current[id];
    if (activeRun?.runId) {
      void api('/api/session/cancel', { sessionId: id, runId: activeRun.runId }).catch(() => {});
      activeRun.controller?.abort();
      streamControllersRef.current[id]?.abort();
      clearRunIfCurrent(id, activeRun.runId);
    }
    if (id === sessionId) {
      stopTtsPlayback({
        clearQueue: true,
        stopVoice: true,
        suppressCurrent: true,
        suppressToken: activeRun?.assistantId ? ttsToken(id, activeRun.assistantId) : '',
      });
      setMessages([]);
      setPendingApproval(null);
      setPendingApprovalAssistantId('');
      setApprovalEdit('');
      setError('');
    }
  }

  function startRenameSession(session) {
    setSessionMenu(null);
    setRenamingSessionId(session.id);
    setRenamingTitle(session.title || session.id);
  }

  function cancelRenameSession() {
    setRenamingSessionId('');
    setRenamingTitle('');
  }

  async function commitRenameSession(session) {
    if (renamingSessionIdRef.current !== session.id) return;
    if (renameSubmittingRef.current === session.id) return;
    const trimmed = renamingTitle.trim();
    if (!trimmed) {
      cancelRenameSession();
      return;
    }
    renameSubmittingRef.current = session.id;
    try {
      const response = await api('/api/session/rename', { sessionId: session.id, title: trimmed });
      if (!response.ok) {
        setError(response.error || t.requestFailed);
        return;
      }
      setSessions(response.sessions || []);
      setDraftSessions((current) =>
        current.map((item) => (item.id === session.id ? { ...item, title: trimmed } : item)),
      );
      if (renamingSessionIdRef.current === session.id) {
        cancelRenameSession();
      }
    } finally {
      if (renameSubmittingRef.current === session.id) {
        renameSubmittingRef.current = '';
      }
    }
  }

  function toggleSessionMenu(session, event) {
    const rect = event.currentTarget.getBoundingClientRect();
    setSessionMenu((open) => {
      if (open?.id === session.id) return null;
      const menuWidth = 148;
      const menuHeight = 122;
      const gutter = 8;
      let left = rect.right - menuWidth;
      let top = rect.bottom + 6;
      if (top + menuHeight > window.innerHeight - gutter) {
        top = rect.top - menuHeight - 6;
      }
      left = Math.max(gutter, Math.min(left, window.innerWidth - menuWidth - gutter));
      top = Math.max(gutter, Math.min(top, window.innerHeight - menuHeight - gutter));
      return { id: session.id, session, left, top };
    });
  }

  function updateComposerInput(value) {
    setInput(value);
    const nextSlashQuery = getSlashQuery(value);
    setSlashMenuOpen(nextSlashQuery !== null && !pendingApproval);
    setSlashMenuIndex(0);
  }

  function resizeComposerTextarea() {
    const textarea = composerTextareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    const maxHeight = 180;
    const nextHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
  }

  function selectSlashCommand(command) {
    setInput(`${command.command} `);
    setSlashMenuOpen(false);
    setSlashMenuIndex(0);
  }

  function handleComposerKeyDown(event) {
    if (slashMenuVisible) {
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        setSlashMenuIndex((index) => (index + 1) % filteredSlashCommands.length);
        return;
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault();
        setSlashMenuIndex((index) => (index - 1 + filteredSlashCommands.length) % filteredSlashCommands.length);
        return;
      }
      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault();
        selectSlashCommand(filteredSlashCommands[slashMenuIndex] || filteredSlashCommands[0]);
        return;
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        setSlashMenuOpen(false);
        return;
      }
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage(event);
    }
  }

  function handleMessageFeedScroll() {
    const feed = scrollRef.current;
    if (!feed) return;
    const nearBottom = isNearScrollBottom(feed);
    stickToBottomRef.current = nearBottom;
    setShowJumpToLatest(!nearBottom);
  }

  function scrollToLatest({ behavior = 'smooth' } = {}) {
    const feed = scrollRef.current;
    if (!feed) return;
    feed.scrollTo({ top: feed.scrollHeight, behavior });
    stickToBottomRef.current = true;
    setShowJumpToLatest(false);
  }

  const statusTone = status?.hasApiKey ? 'ready' : 'missing';
  const lastMessage = messages[messages.length - 1];
  const latestProgressAssistantId = findLastProgressAssistantId(messages);
  const showBusyPlaceholder = interactionBusy && !approvalProcessing && lastMessage?.role !== 'assistant';

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      {sidebarCollapsed && (
        <button
          className="sidebar-reopen"
          onClick={() => setSidebarCollapsed(false)}
          aria-label={t.showSidebar}
          title={t.showSidebar}
        >
          <PanelLeftOpen size={18} />
        </button>
      )}
      <aside className="sidebar" aria-label="Session navigation" aria-hidden={sidebarCollapsed} inert={sidebarCollapsed}>
        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-mark">
              <Code2 size={18} />
            </div>
            <div>
              <strong>LangCode</strong>
              <span>{t.codeAgent}</span>
            </div>
          </div>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed(true)}
            aria-label={t.hideSidebar}
            title={t.hideSidebar}
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        <button className="new-session" onClick={newSession} disabled={busy || creatingSession}>
          <MessageSquarePlus size={17} />
          {t.newSession}
        </button>

        <div className="session-stack">
          <p className="nav-label">{t.chats}</p>
          {workspaceGroups.map((group) => {
            const collapsed = Boolean(collapsedWorkspaces[group.workspace]);
            return (
              <section className="workspace-group" key={group.workspace || 'unbound'}>
                <button className="workspace-group-header" onClick={() => toggleWorkspaceGroup(group.workspace)}>
                  {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
                  <Folder size={17} />
                  <span>{group.name}</span>
                </button>
                {!collapsed && (
                  <div className="workspace-session-list">
                    {group.sessions.map((session) => (
                      <div key={session.id} className={`session-row ${session.id === sessionId ? 'active' : ''}`}>
                        {renamingSessionId === session.id ? (
                          <div className="session-item session-editing">
                            <Bot size={16} />
                            <input
                              ref={renameInputRef}
                              value={renamingTitle}
                              onChange={(event) => setRenamingTitle(event.target.value)}
                              onBlur={() => commitRenameSession(session)}
                              onKeyDown={(event) => {
                                if (event.key === 'Enter') {
                                  event.preventDefault();
                                  commitRenameSession(session);
                                }
                                if (event.key === 'Escape') {
                                  event.preventDefault();
                                  cancelRenameSession();
                                }
                              }}
                              aria-label={t.renameSessionPrompt}
                            />
                          </div>
                        ) : (
                          <button className="session-item" onClick={() => openSession(session.id)}>
                            <Bot size={16} />
                            {runningSessions[session.id] && <span className="session-running-dot" aria-label={t.working} />}
                            {!runningSessions[session.id] && unreadSessions[session.id] && (
                              <span className="session-unread-dot" aria-label={t.sessionCompleted} />
                            )}
                            <span>{session.title || session.id}</span>
                          </button>
                        )}
                        <button
                          className="session-menu-button"
                          onClick={(event) => toggleSessionMenu(session, event)}
                          aria-label={t.renameSession}
                        >
                          <MoreHorizontal size={17} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            );
          })}
        </div>

        <div className="sidebar-bottom">
          {settingsOpen && (
            <form className="settings-panel" onSubmit={saveSettings}>
              <label>
                <span>{t.model}</span>
                <select value={modelInput} onChange={(event) => setModelInput(event.target.value)}>
                  {modelOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              {selectedSupportsThinking && (
                <label className="settings-check-row">
                  <input
                    type="checkbox"
                    checked={thinkingEnabled}
                    onChange={(event) => setThinkingEnabled(event.target.checked)}
                  />
                  <span>{t.thinkingMode}</span>
                  <small>{t.thinkingModeHint}</small>
                </label>
              )}
              <label>
                <span>{t.voiceInputModel}</span>
                <select value={voiceModel} onChange={(event) => setVoiceModel(event.target.value)}>
                  <option value="qwen3-asr-0.6b">{t.voiceModelQwen}</option>
                </select>
              </label>
              <label className="settings-check-row">
                <input type="checkbox" checked={ttsEnabled} onChange={(event) => setTtsEnabled(event.target.checked)} />
                <span>{t.ttsEnabled}</span>
                <small>默认音色 / 本地音色 / 语音打断</small>
              </label>
              <label>
                <span>{t.ttsVoice}</span>
                <div className="voice-select-row">
                  <select value={ttsVoiceId} onChange={(event) => setTtsVoiceId(event.target.value)}>
                    {ttsVoiceOptions.map((voice) => (
                      <option key={voice.id} value={voice.id}>
                        {voice.name || voice.id}
                      </option>
                    ))}
                  </select>
                  <button
                    className="voice-preview-button"
                    type="button"
                    title={t.previewVoice}
                    aria-label={t.previewVoice}
                    disabled={!ttsVoiceOptions.find((voice) => voice.id === ttsVoiceId)?.previewUrl}
                    onClick={playVoicePreview}
                  >
                    <AudioLines size={16} />
                  </button>
                </div>
                {ttsVoiceOptions.find((voice) => voice.id === ttsVoiceId)?.previewText && (
                  <small>{ttsVoiceOptions.find((voice) => voice.id === ttsVoiceId)?.previewText}</small>
                )}
              </label>
              <label>
                <span>{t.displayLanguage}</span>
                <select value={language} onChange={(event) => setLanguage(event.target.value)}>
                  <option value="zh">{t.chinese}</option>
                  <option value="en">{t.english}</option>
                </select>
              </label>
              <button className="settings-save" type="submit" disabled={workspaceBusy}>
                {t.saveSettings}
              </button>
            </form>
          )}

          <div className="sidebar-model-panel">
            <div className="model-status-row">
              <div className={`model-status-dot ${statusTone}`} aria-hidden="true" />
              <div className="meta-block">
                <span>{t.provider}</span>
                <strong>{status?.provider || t.loading}</strong>
                <small>{status?.model || t.modelPending}</small>
              </div>
              <span className={`status-pill ${statusTone}`}>
                {status?.hasApiKey ? t.keyReady : t.toolMode}
              </span>
            </div>
            <div className="theme-toggle" role="group" aria-label={t.theme}>
              {[
                { mode: 'system', icon: <Monitor size={14} />, label: t.themeSystem },
                { mode: 'light', icon: <Sun size={14} />, label: t.themeLight },
                { mode: 'dark', icon: <Moon size={14} />, label: t.themeDark },
              ].map((option) => (
                <button
                  key={option.mode}
                  type="button"
                  className={themeMode === option.mode ? 'active' : ''}
                  aria-pressed={themeMode === option.mode}
                  title={option.label}
                  aria-label={option.label}
                  onClick={() => applyThemeMode(option.mode)}
                >
                  {option.icon}
                </button>
              ))}
            </div>
            <button
              className={`settings-button ${settingsOpen ? 'active' : ''}`}
              onClick={() => setSettingsOpen((open) => !open)}
              aria-expanded={settingsOpen}
            >
              <Settings size={17} />
              {t.settings}
            </button>
          </div>
        </div>
      </aside>

      {!sidebarCollapsed && (
        <button
          type="button"
          className="sidebar-scrim"
          onClick={() => setSidebarCollapsed(true)}
          aria-label={t.close}
        />
      )}

      <main className="conversation" aria-label="Chat workspace">
        <header className="topbar">
          <div>
            <p>LangCode</p>
            <strong>{t.chatHeader}</strong>
          </div>
          <div className="topbar-status">
            <span>{t.workspaceReady}</span>
            <strong>{status?.model || t.modelPending}</strong>
          </div>
        </header>

        <section
          className="message-feed"
          ref={scrollRef}
          onScroll={handleMessageFeedScroll}
          role="log"
          aria-live="polite"
          aria-relevant="additions"
        >
          <div className="message-column">
            {messages.map((message) => (
              <Message
                key={message.id}
                message={message}
                t={t}
                showProgress={message.id === latestProgressAssistantId}
              />
            ))}
            {showBusyPlaceholder && (
              <div className="message assistant pulse">
                <div className="bubble">{t.working}</div>
              </div>
            )}
          </div>
        </section>

        <div className="toast-stack" role="status" aria-live="polite">
          {error && (
            <div className="toast">
              <CircleAlert size={16} />
              <span>{error}</span>
              <button type="button" onClick={() => setError('')} aria-label={t.close}><X size={15} /></button>
            </div>
          )}
          {voiceError && (
            <div className="toast">
              <CircleAlert size={16} />
              <span>{voiceError}</span>
              <button type="button" onClick={() => setVoiceError('')} aria-label={t.close}><X size={15} /></button>
            </div>
          )}
          {toasts.map((toast) => (
            <div key={toast.id} className={`toast ${toast.tone === 'info' ? 'info' : ''}`.trim()}>
              <CircleAlert size={16} />
              <span>{toast.message}</span>
              {toast.retry?.text && (
                <button type="button" className="toast-retry" onClick={() => void retryFailedTurn(toast)}>
                  {t.retry}
                </button>
              )}
              <button type="button" onClick={() => dismissToast(toast.id)} aria-label={t.close}><X size={15} /></button>
            </div>
          ))}
        </div>

        <div className="composer-stack">
          {showJumpToLatest && (
            <button type="button" className="jump-latest" onClick={() => scrollToLatest()} aria-label={t.jumpToLatest}>
              <ArrowDown size={15} />
              {t.jumpToLatest}
            </button>
          )}

          {pendingApproval && (
            <section
              className="approval-strip"
              aria-label="Tool approval"
              ref={approvalSectionRef}
              tabIndex={-1}
              onKeyDown={(event) => {
                if (event.key === 'Escape') {
                  event.preventDefault();
                  const target = event.target;
                  // Esc inside the edit box only leaves the box; a second Esc rejects.
                  if (target instanceof HTMLTextAreaElement || target instanceof HTMLInputElement) {
                    target.blur();
                    approvalSectionRef.current?.focus();
                    return;
                  }
                  approve('reject');
                } else if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault();
                  approve('accept');
                }
              }}
            >
              <div className="approval-summary">
                <SlidersHorizontal size={16} />
                <div>
                  <strong>{format(t.allowTool, { tool: pendingApproval.toolName })}</strong>
                  <span>{pendingApproval.payload?.risk?.reason || t.reviewToolInput}</span>
                </div>
              </div>
              <ApprovalToolInput toolName={pendingApproval.toolName} toolInput={pendingApproval.toolInput} t={t} />
              <textarea
                value={approvalEdit}
                onChange={(event) => setApprovalEdit(event.target.value)}
                className={approvalEditError ? 'invalid' : ''}
                rows={3}
                aria-label={t.approvalEditLabel}
                aria-invalid={Boolean(approvalEditError)}
                aria-describedby={approvalEditError ? 'approval-json-error' : undefined}
              />
              {approvalEditError && <span id="approval-json-error" className="approval-json-error">{approvalEditError}</span>}
              <div className="approval-actions">
                <button className="accept" onClick={() => approve('accept')}>
                  <Check size={15} />
                  {t.accept}
                </button>
                {pendingApproval.toolName === 'shell' && (
                  <button className="accept-secondary" onClick={() => approve('accept', { remember: true })}>
                    <Check size={15} />
                    {t.acceptRemember}
                  </button>
                )}
                <button disabled={Boolean(approvalEditError)} onClick={() => approve('edit')}>{t.edit}</button>
                <button onClick={() => approve('feedback')}>{t.feedback}</button>
                <button className="reject" onClick={() => approve('reject')}>
                  <X size={15} />
                  {t.reject}
                </button>
                <button className="force-end" onClick={forceEndConversation}>
                  <Square size={14} fill="currentColor" />
                  {t.forceEnd}
                </button>
              </div>
            </section>
          )}

          {slashMenuVisible && (
            <section className="slash-menu" role="listbox" aria-label={t.slashCommands}>
              <div className="slash-menu-header">
                <strong>{t.slashCommands}</strong>
                <span>{t.slashCommandHint}</span>
              </div>
              {filteredSlashCommands.map((item, index) => (
                <button
                  key={item.command}
                  type="button"
                  className={index === slashMenuIndex ? 'active' : ''}
                  role="option"
                  aria-selected={index === slashMenuIndex}
                  onMouseEnter={() => setSlashMenuIndex(index)}
                  onMouseDown={(event) => {
                    event.preventDefault();
                    selectSlashCommand(item);
                  }}
                >
                  <code>{item.command}</code>
                  <span>{item.labels[language]}</span>
                  <small>{item.descriptions[language]}</small>
                </button>
              ))}
            </section>
          )}

          {(voiceActive || voiceConversationActive) && (
            <section className="voice-strip" aria-label={t.voiceFeature}>
              <div className="voice-orb" aria-hidden="true">
                <AudioLines size={17} />
              </div>
              <div>
                <strong>{voiceStatus || t.voiceListening}</strong>
                <span>{input || t.voiceReady}</span>
              </div>
            </section>
          )}

          <form className="composer" onSubmit={sendMessage}>
            <div className={`composer-input-shell ${selectedSlashCommand ? 'has-highlight' : ''}`}>
              <ComposerHighlight input={input} selectedCommand={selectedSlashCommand?.command} />
              <textarea
                ref={composerTextareaRef}
                value={input}
                onChange={(event) => updateComposerInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder={!sessionId ? t.chooseWorkspaceFirst : pendingApproval ? t.pendingPlaceholder : t.askPlaceholder}
                rows={1}
                disabled={!sessionId || Boolean(pendingApproval) || voiceActive}
                spellCheck={false}
              />
            </div>
            <button
              type="button"
              className={`voice-button ${voiceActive || voiceConversationActive ? 'active' : ''}`}
              disabled={!sessionId || (Boolean(pendingApproval) && !voiceActive && !voiceConversationActive)}
              onClick={
                voiceActive || voiceConversationActive
                  ? stopVoiceConversation
                  : startVoiceConversation
              }
              aria-label={voiceActive || voiceConversationActive ? t.stopVoice : activeSessionBusy ? t.voiceInterrupt : t.voiceFeature}
              title={voiceActive || voiceConversationActive ? t.stopVoice : activeSessionBusy ? t.voiceInterrupt : t.voiceFeature}
            >
              <AudioLines size={18} />
            </button>
            <button
              type={activeSessionBusy ? 'button' : 'submit'}
              className={activeSessionBusy ? 'stop' : ''}
              disabled={
                !sessionId ||
                busy ||
                (!activeSessionBusy && (voiceActive || !input.trim() || Boolean(pendingApproval)))
              }
              onClick={activeSessionBusy ? stopGeneration : undefined}
              aria-label={activeSessionBusy ? t.stopGeneration : t.sendMessage}
              title={activeSessionBusy ? t.stopGeneration : t.sendMessage}
            >
              {activeSessionBusy ? <Square size={15} fill="currentColor" /> : <Send size={18} />}
            </button>
            {pendingApproval && !activeSessionBusy && (
              <button
                type="button"
                className="force-stop"
                onClick={forceEndConversation}
                aria-label={t.forceEnd}
                title={t.forceEnd}
              >
                <Square size={15} fill="currentColor" />
              </button>
            )}
          </form>

          <div className="session-workspace-bar" aria-label={t.sessionWorkspace}>
            <Folder size={16} />
            <span>{t.sessionWorkspace}</span>
            <strong>{activeWorkspaceLabel || t.chooseWorkspaceFirst}</strong>
          </div>
        </div>
      </main>
      {sessionMenu &&
        createPortal(
            <div className="session-menu" style={{ left: sessionMenu.left, top: sessionMenu.top }}>
              <button className="rename-action" onClick={() => startRenameSession(sessionMenu.session)}>
                {t.renameSession}
              </button>
              <button className="clear-action" onClick={() => clearSession(sessionMenu.id)}>
                {t.clearSession}
              </button>
            <button className="delete-action" onClick={() => deleteSession(sessionMenu.id)}>
              {t.deleteSession}
            </button>
          </div>,
          document.body,
        )}
    </div>
  );
}

function getSlashQuery(value) {
  if (!value.startsWith('/')) return null;
  if (value.includes('\n')) return null;
  const firstToken = value.split(/\s/)[0];
  if (firstToken !== value) return null;
  return firstToken.slice(1).toLowerCase();
}

function isNearScrollBottom(element, threshold = 120) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= threshold;
}

function appendResponseMessages(currentMessages, responseMessages, targetAssistantId = '') {
  const nextMessages = [...currentMessages];
  const pendingAssistantIndex = targetAssistantId
    ? nextMessages.findIndex((message) => message.id === targetAssistantId)
    : nextMessages.length > 0 &&
        nextMessages[nextMessages.length - 1].role === 'assistant' &&
        nextMessages[nextMessages.length - 1].kind === 'message' &&
        !String(nextMessages[nextMessages.length - 1].content || '').trim()
      ? nextMessages.length - 1
      : -1;

  const assistantContents = [];
  const passthroughMessages = [];
  for (const message of responseMessages) {
    if (message.role === 'assistant' && message.kind === 'message') {
      const content = String(message.content || '').trim();
      if (content) assistantContents.push(content);
      continue;
    }
    if (message.role === 'tool' && message.kind === 'tool_result' && shouldHideToolResult(message.content)) {
      continue;
    }
    if (message.kind === 'agent_dialogue') {
      passthroughMessages.push({
        ...message,
        agentMessages: Array.isArray(message.agentMessages) ? message.agentMessages : message.messages || [],
      });
      continue;
    }
    passthroughMessages.push(message);
  }

  if (assistantContents.length) {
    const mergedAssistant = {
      role: 'assistant',
      kind: 'message',
      content: assistantContents.join('\n\n'),
    };
    if (pendingAssistantIndex >= 0) {
      const nextContent = joinAssistantContent(nextMessages[pendingAssistantIndex].content, mergedAssistant.content);
      nextMessages[pendingAssistantIndex] = {
        ...nextMessages[pendingAssistantIndex],
        role: 'assistant',
        kind: 'message',
        content: nextContent,
        contentPlacement: contentPlacementForMessage({
          ...nextMessages[pendingAssistantIndex],
          content: nextContent,
        }),
      };
    } else {
      nextMessages.push({ ...mergedAssistant, id: crypto.randomUUID() });
    }
  }

  nextMessages.push(...passthroughMessages.map((message) => ({ ...message, id: crypto.randomUUID() })));
  return nextMessages;
}

function upsertAgentDialogueMessage(messages, event) {
  const threadId = String(event.threadId || '');
  const nextMessage = {
    id: crypto.randomUUID(),
    role: 'assistant',
    kind: 'agent_dialogue',
    title: event.title || 'Agent 协作',
    dialogueType: event.dialogueType || 'agent_dialogue',
    threadId,
    participants: Array.isArray(event.participants) ? event.participants : [],
    agentMessages: Array.isArray(event.messages) ? event.messages : [],
  };
  if (!threadId) return [...messages, nextMessage];
  let found = false;
  const updated = messages.map((message) => {
    if (message.kind !== 'agent_dialogue' || message.threadId !== threadId) return message;
    found = true;
    return {
      ...message,
      title: nextMessage.title,
      dialogueType: nextMessage.dialogueType,
      participants: nextMessage.participants,
      agentMessages: nextMessage.agentMessages,
    };
  });
  return found ? updated : [...updated, nextMessage];
}

function ensurePostToolAssistantMessage(messages, run) {
  if (!run) return messages;
  if (run.responseAssistantId && messages.some((message) => message.id === run.responseAssistantId)) {
    return messages;
  }
  const postToolAssistantId = run.responseAssistantId || crypto.randomUUID();
  run.responseAssistantId = postToolAssistantId;
  return [
    ...messages,
    {
      id: postToolAssistantId,
      role: 'assistant',
      kind: 'message',
      content: '',
      thinking: '',
      thinkingRunning: false,
      progress: emptyProgress(),
      todos: [],
      progressRunning: false,
    },
  ];
}

function emptyProgress() {
  return { items: [], current: null, summary: '' };
}

function findLastAssistantIndex(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'assistant') return index;
  }
  return -1;
}

function findLastAssistantId(messages) {
  const index = findLastAssistantIndex(messages);
  return index >= 0 ? messages[index].id : '';
}

function findLastProgressAssistantId(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === 'assistant' && hasAssistantProgress(message)) {
      return message.id;
    }
  }
  return '';
}

function joinAssistantContent(existing, incoming) {
  const current = String(existing || '').trim();
  const next = String(incoming || '').trim();
  if (!current) return next;
  if (!next) return current;
  return `${current}\n\n${next}`;
}

function attachTodosToLastAssistant(messages, todos) {
  if (!Array.isArray(todos) || todos.length === 0) return messages;
  const index = findLastAssistantIndex(messages);
  if (index < 0) return messages;
  const next = [...messages];
  const message = next[index];
  next[index] = {
    ...message,
    todos,
    progress: message.progress || emptyProgress(),
    progressRunning: false,
    contentPlacement: contentPlacementForMessage({ ...message, todos, progress: message.progress || emptyProgress() }),
  };
  return next;
}

function shouldAttachSessionTodos(messages, todos, pendingApproval) {
  if (pendingApproval) return true;
  return false;
}

function updateAssistantProgressInMessages(messages, assistantId, updater) {
  return messages.map((message) => {
    if (message.id !== assistantId) return message;
    const progress = message.progress || emptyProgress();
    const nextMessage = { ...message, progress: updater(progress) };
    nextMessage.contentPlacement = contentPlacementForMessage(nextMessage);
    return nextMessage;
  });
}

function setAssistantTodosInMessages(messages, assistantId, nextTodos) {
  return messages.map((message) =>
    message.id === assistantId
      ? {
          ...message,
          todos: nextTodos,
          progress: message.progress || emptyProgress(),
          contentPlacement: contentPlacementForMessage({
            ...message,
            todos: nextTodos,
            progress: message.progress || emptyProgress(),
          }),
        }
      : message,
  );
}

function setAssistantProgressRunningInMessages(messages, assistantId, running) {
  return messages.map((message) => (message.id === assistantId ? { ...message, progressRunning: running } : message));
}

function appendAssistantThinkingInMessages(messages, assistantId, text) {
  if (!text) return messages;
  return messages.map((message) =>
    message.id === assistantId
      ? {
          ...message,
          thinking: `${message.thinking || ''}${text}`,
          thinkingRunning: true,
        }
      : message,
  );
}

function appendAssistantContentInMessages(messages, assistantId, content) {
  const value = String(content || '');
  if (!value) return messages;
  return messages.map((message) => {
    if (message.id !== assistantId) return message;
    const nextContent = `${message.content || ''}${value}`;
    return {
      ...message,
      content: nextContent,
      thinkingRunning: false,
      contentPlacement: contentPlacementForMessage({ ...message, content: nextContent }),
    };
  });
}

function setAssistantHeartbeatInMessages(messages, assistantId, waitedSec) {
  let changed = false;
  const next = messages.map((message) => {
    if (message.id !== assistantId) return message;
    const current = message.heartbeatSec ?? null;
    if (current === waitedSec) return message;
    changed = true;
    return { ...message, heartbeatSec: waitedSec };
  });
  return changed ? next : messages;
}

function setAssistantUsageInMessages(messages, assistantId, usage) {
  return messages.map((message) => (message.id === assistantId ? { ...message, usage } : message));
}

function attachToolPreviewToProgress(progress, event) {
  const items = progress?.items || [];
  if (!items.length) return progress;
  let index = items.length - 1;
  if (event.toolName) {
    for (let cursor = items.length - 1; cursor >= 0; cursor -= 1) {
      if (items[cursor].toolName === event.toolName) {
        index = cursor;
        break;
      }
    }
  }
  const nextItems = [...items];
  nextItems[index] = {
    ...nextItems[index],
    preview: String(event.preview || ''),
    previewTruncated: Boolean(event.truncated),
  };
  return { ...progress, items: nextItems };
}

const ERROR_CODE_TRANSLATION_KEYS = {
  auth: 'errorAuth',
  rate_limit: 'errorRateLimit',
  model_timeout: 'errorModelTimeout',
  context_overflow: 'errorContextOverflow',
  network: 'errorNetwork',
};

function errorCodeHint(code, t) {
  const key = ERROR_CODE_TRANSLATION_KEYS[String(code || '')];
  return key ? t[key] || '' : '';
}

function archiveAssistantProgress(messages) {
  return messages.map((message) => {
    if (message.role !== 'assistant') return message;
    if (!hasAssistantProgress(message)) {
      return message.voiceTtsPreparing ? { ...message, voiceTtsPreparing: false } : message;
    }
    return {
      ...message,
      progress: emptyProgress(),
      todos: [],
      progressRunning: false,
      progressArchived: true,
      voiceTtsPreparing: false,
    };
  });
}

function hasAssistantProgress(message) {
  const progress = message.progress || emptyProgress();
  return Boolean(
    progress.current ||
      (progress.items?.length || 0) > 0 ||
      progress.summary ||
      (message.todos?.length || 0) > 0,
  );
}

function hasVisibleAssistantOutput(message) {
  if (!message || message.role !== 'assistant') return false;
  if (String(message.content || '').trim()) return true;
  if (String(message.thinking || '').trim()) return true;
  const progress = message.progress || emptyProgress();
  if (progress.current || (progress.items?.length || 0) > 0 || (message.todos?.length || 0) > 0) return true;
  return false;
}

function shouldDiscardInterruptedRun(messages, activeRun) {
  if (!activeRun?.assistantId || !activeRun?.userMessageId) return false;
  const assistant = messages.find((message) => message.id === activeRun.assistantId);
  return !hasVisibleAssistantOutput(assistant);
}

function findPreviousUserMessage(messages, assistantId) {
  const assistantIndex = messages.findIndex((message) => message.id === assistantId);
  for (let index = (assistantIndex >= 0 ? assistantIndex : messages.length) - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'user') return messages[index];
  }
  return null;
}

function stripMarkdownForSpeech(text) {
  const lines = String(text || '')
    .split(/\n+/)
    .map((line) => line.trim())
    .filter((line) => line && !/^\|?\s*:?-{3,}/.test(line))
    .map((line) => line.replace(/\|/g, '，'));
  return lines
    .join('。')
    .replace(/```[\s\S]*?```/g, ' 代码块 ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/https?:\/\/\S+/g, ' 网页链接 ')
    .replace(/^\s*#{1,6}\s*/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+[.)、]\s+/gm, '')
    .replace(/[#>*_\-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 1600);
}

function ttsToken(sessionId, assistantId) {
  return `${sessionId}:${assistantId}`;
}

function parseTtsToken(token) {
  const value = String(token || '');
  const separator = value.lastIndexOf(':');
  if (separator <= 0 || separator >= value.length - 1) return null;
  return {
    sessionId: value.slice(0, separator),
    assistantId: value.slice(separator + 1),
  };
}

function splitTtsText(text, { force = false, compact = false, first = false } = {}) {
  let buffer = String(text || '').trimStart();
  const chunks = [];
  let firstChunk = first;
  const readyLength = () => (firstChunk ? FIRST_TTS_CHUNK_MIN : compact ? 72 : 48);
  while (buffer.length >= readyLength()) {
    const boundary = findBalancedTtsBoundary(buffer, { compact, first: firstChunk });
    if (boundary <= 0) break;
    chunks.push(buffer.slice(0, boundary).trim());
    buffer = buffer.slice(boundary).trimStart();
    firstChunk = false;
  }
  if (force && buffer.trim()) {
    appendFinalTtsChunks(chunks, splitBalancedFinalTtsChunks(buffer.trim(), { compact }), { compact });
    buffer = '';
  }
  return { chunks: chunks.filter(Boolean), remainder: buffer };
}

function appendFinalTtsChunks(chunks, finalChunks, { compact = false } = {}) {
  const minTail = compact ? 36 : 24;
  for (const chunk of finalChunks) {
    if (chunk.length < minTail && chunks.length > 0) {
      const previous = chunks[chunks.length - 1] || '';
      const joiner = previous && /[a-zA-Z0-9]$/.test(previous) && /^[a-zA-Z0-9]/.test(chunk) ? ' ' : '';
      chunks[chunks.length - 1] = `${previous}${joiner}${chunk}`;
    } else {
      chunks.push(chunk);
    }
  }
}

// The very first spoken chunk is cut short (12-20 chars, or at the first sentence
// punctuation, whichever comes first) so playback starts as soon as possible.
const FIRST_TTS_CHUNK_MIN = 12;
const FIRST_TTS_CHUNK_MAX = 20;
const STRONG_TTS_MARKS = ['。', '！', '？', '!', '?', '；', ';', '\n'];
const WEAK_TTS_MARKS = ['，', ',', '、', '：', ':'];

function findFirstTtsBoundary(text) {
  const length = text.length;
  if (length < FIRST_TTS_CHUNK_MIN) return -1;
  const endLimit = Math.min(length, FIRST_TTS_CHUNK_MAX);
  const strong = earliestTtsCut(text, { to: endLimit, marks: STRONG_TTS_MARKS });
  if (strong > 0) return strong;
  const weak = earliestTtsCut(text, { from: FIRST_TTS_CHUNK_MIN, to: endLimit, marks: WEAK_TTS_MARKS });
  if (weak > 0) return weak;
  if (length >= FIRST_TTS_CHUNK_MAX) return snapToWordBoundary(text, endLimit);
  return -1;
}

function earliestTtsCut(text, { from = 1, to, marks }) {
  const markSet = new Set(marks);
  for (let index = Math.max(0, from - 1); index < to; index += 1) {
    if (!markSet.has(text[index])) continue;
    let end = index + 1;
    while (end < text.length && '”’）】」』'.includes(text[end])) end += 1;
    return end;
  }
  return -1;
}

function findBalancedTtsBoundary(text, { compact = false, first = false } = {}) {
  if (first) return findFirstTtsBoundary(text);
  if (compact) return findCustomVoiceTtsBoundary(text);
  const length = text.length;
  const min = 44;
  const target = 66;
  const max = 88;
  if (length < min) return -1;
  const endLimit = Math.min(length, max);
  const preferred = nearestTtsCut(text, { from: min, target, to: endLimit });
  if (preferred > 0) return preferred;
  if (length >= max) return snapToWordBoundary(text, endLimit);
  return -1;
}

function findCustomVoiceTtsBoundary(text) {
  const length = text.length;
  const min = 64;
  const target = 92;
  const max = 120;
  if (length < min) return -1;
  const endLimit = Math.min(length, max);
  const strong = nearestTtsCut(text, {
    from: min,
    target,
    to: endLimit,
    marks: STRONG_TTS_MARKS,
  });
  if (strong > 0) return strong;
  if (length >= target) {
    const weak = nearestTtsCut(text, {
      from: Math.max(min, target - 10),
      target,
      to: endLimit,
      marks: WEAK_TTS_MARKS,
    });
    if (weak > 0) return weak;
  }
  if (length >= max) return snapToWordBoundary(text, endLimit);
  return -1;
}

function nearestTtsCut(text, { from, target, to, marks: allowedMarks = null }) {
  const marks = new Set(allowedMarks || ['。', '！', '？', '!', '?', '；', ';', '\n', '，', ',', '、', '：', ':']);
  let best = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = from - 1; index < to; index += 1) {
    if (!marks.has(text[index])) continue;
    let end = index + 1;
    while (end < text.length && '”’）】」』'.includes(text[end])) end += 1;
    const distance = Math.abs(end - target);
    if (distance < bestDistance) {
      best = end;
      bestDistance = distance;
    }
  }
  return best;
}

function splitBalancedFinalTtsChunks(text, { compact = false } = {}) {
  let buffer = String(text || '').trim();
  const chunks = [];
  const max = compact ? 76 : 88;
  const minTail = compact ? 18 : 24;
  while (buffer.length > max) {
    if (compact && buffer.length <= max + 8) break;
    const end = findBalancedTtsBoundary(buffer, { compact }) || snapToWordBoundary(buffer, max);
    chunks.push(buffer.slice(0, end).trim());
    buffer = buffer.slice(end).trimStart();
  }
  if (buffer) {
    if (chunks.length > 0 && buffer.length < minTail) {
      const previous = chunks[chunks.length - 1] || '';
      const joiner = previous && /[a-zA-Z0-9]$/.test(previous) && /^[a-zA-Z0-9]/.test(buffer) ? ' ' : '';
      chunks[chunks.length - 1] = `${previous}${joiner}${buffer}`;
    } else {
      chunks.push(buffer);
    }
  }
  return chunks;
}

function snapToWordBoundary(text, fallbackEnd) {
  const end = Math.min(String(text || '').length, fallbackEnd);
  for (let index = end; index > Math.max(0, end - 8); index -= 1) {
    if (/\s/.test(text[index] || '')) return index + 1;
  }
  return end;
}

function base64AudioBlob(value, contentType) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: contentType || 'audio/wav' });
}

const BARGE_IN_CJK_FILLERS = ['嗯', '啊', '哦', '对', '好'];
const BARGE_IN_LATIN_FILLERS = ['uh', 'um', 'yeah', 'ok'];

// Non-keyword barge-in: enough real content to be a genuine interruption rather than a
// backchannel. Chinese is counted in characters, English in words.
function isSubstantialBargeInSpeech(text) {
  let stripped = String(text || '')
    .toLowerCase()
    .replace(/[，,。！？!?；;：:、~～…"'“”‘’（）()]/g, ' ');
  for (const filler of BARGE_IN_CJK_FILLERS) stripped = stripped.split(filler).join(' ');
  const words = stripped
    .split(/\s+/)
    .filter((word) => word && /[a-z0-9]/.test(word) && !BARGE_IN_LATIN_FILLERS.includes(word));
  if (words.length >= 2) return true;
  return (stripped.match(/[一-鿿]/g) || []).length >= 4;
}

function isBargeInIntent(text) {
  const compact = String(text || '').replace(/\s+/g, '').toLowerCase();
  if (!compact) return false;
  const keywords = [
    '停停',
    '停一下',
    '停下来',
    '先停',
    '先停下',
    '暂停',
    '停止播报',
    '别说了',
    '别讲了',
    '不要说了',
    '不用说了',
    '先别说',
    '打断一下',
    '等一下',
    '等等',
    '不对',
    '不对不对',
    '这个不对',
    '你说的不对',
    '方向不对',
    '不是这样',
    '我觉得不是这样',
    '不是这个意思',
    '我不是这个意思',
    '换个说法',
    '重新说',
    '先听我说',
    '停',
    '停止',
    '别播了',
    '别念了',
    '不用播了',
    '打断',
    '闭嘴',
    '错了',
    // English interrupts; `compact` is whitespace-free and lower-cased.
    'stop',
    'wait',
    'holdon',
    'nono',
    "that'swrong",
    'thatswrong',
    'nevermind',
  ];
  return keywords.some((keyword) => compact.includes(keyword));
}

function composeBargeInToolInput(interruptText, context) {
  const userText = String(context?.userText || '').trim();
  const assistantText = String(context?.assistantText || '').trim();
  const spoken = String(interruptText || '').trim();
  if (!spoken) return null;
  return {
    spokenText: spoken,
    previousUserText: userText,
    assistantDisplayedText: assistantText,
  };
}

function contentPlacementForMessage(message) {
  return hasAssistantProgress(message) ? 'afterProgress' : 'beforeProgress';
}

function ComposerHighlight({ input, selectedCommand }) {
  if (!input || !selectedCommand) return null;
  return (
    <div className="composer-highlight" aria-hidden="true">
      <span className="slash-token">{selectedCommand}</span>
      {input.slice(selectedCommand.length) || '\u00a0'}
    </div>
  );
}

function updateAgentProgress(current, event, t) {
  if (event.status === 'summary') {
    return {
      ...current,
      current: null,
      summary: event.label || t.progressNext,
    };
  }

  const nextItem = {
    id: `${event.toolName || 'tool'}-${event.step || 0}-${event.status}`,
    status: event.status,
    toolName: event.toolName || 'tool',
    target: event.target || '',
    step: event.step || 0,
    total: event.total || 0,
    ok: event.ok,
    label: progressLabel(event, t),
  };

  if (event.status === 'running' || event.status === 'waiting_approval') {
    return { ...current, current: nextItem };
  }

  if (event.status === 'completed') {
    return {
      ...current,
      current: null,
      summary: format(t.progressCompleted, { count: event.completed || current.items.length + 1 }),
      items: [...current.items, nextItem],
    };
  }

  return current;
}

function finishAgentProgress(current, t) {
  if (!current.current) {
    return current.items.length ? current : { ...current, summary: '' };
  }
  return {
    ...current,
    current: null,
    summary: current.items.length ? format(t.progressCompleted, { count: current.items.length }) : '',
  };
}

function shouldHideToolResult(content) {
  const text = String(content || '').trim();
  if (!text) return true;
  try {
    return isSuccessfulToolResult(JSON.parse(text));
  } catch {
    return false;
  }
}

function isSuccessfulToolResult(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  return value.ok === true;
}

function progressLabel(event, t) {
  if (event.status === 'waiting_approval') {
    return `${t.progressWaitingApproval}: ${toolActionLabel(event.toolName, t)} ${event.target || ''}`.trim();
  }
  if (event.label) return event.label;
  return `${toolActionLabel(event.toolName, t)} ${event.target || ''}`.trim();
}

function toolActionLabel(toolName, t) {
  return t.progressActions?.[toolName] || format(t.processingTool, { tool: toolName || 'tool' });
}

function isTaskPlanningTool(toolName) {
  return ['task_create', 'task_update', 'task_list', 'task_get', 'task_cancel'].includes(toolName);
}

function todoStatusForDisplay(item, index, todos, progress, running) {
  const status = item.status || 'pending';
  if (status !== 'pending') return status;
  const hasExplicitActive = todos.some((todo) => todo.status === 'in_progress');
  const hasExplicitDone = todos.some((todo) => todo.status === 'completed');
  const currentIsWork = progress.current && !isTaskPlanningTool(progress.current.toolName);
  const hasCompletedWork = progress.items.some((item) => !isTaskPlanningTool(item.toolName));
  if (running && !hasExplicitActive && !hasExplicitDone && (currentIsWork || hasCompletedWork) && index === 0) {
    return 'in_progress';
  }
  return status;
}

function statusIcon(status) {
  if (status === 'completed') return '✓';
  if (status === 'in_progress') return '';
  if (status === 'blocked') return '!';
  if (status === 'cancelled') return '×';
  return '';
}

function AgentProgress({ progress, todos, t, running }) {
  const hasTodos = Array.isArray(todos) && todos.length > 0;
  const hasProgress = Boolean(progress.current || progress.items.length > 0 || progress.summary || hasTodos);
  if (!hasProgress) return null;
  const completedCount = hasTodos ? todos.filter((item) => item.status === 'completed').length : progress.items.length;
  return (
    <section className="agent-progress" aria-label={t.progressTitle}>
      <div className="agent-progress-header">
        <span>{t.progressTitle}</span>
        <strong>{format(t.progressCompleted, { count: progress.items.length })}</strong>
      </div>
      {hasTodos && (
        <div className="agent-todos">
          <div className="agent-todos-heading">
            <strong>{t.planTitle}</strong>
            <span>
              {completedCount}/{todos.length}
            </span>
          </div>
          <ol>
            {todos.map((item, index) => {
              const status = todoStatusForDisplay(item, index, todos, progress, running);
              const animateTodo = shouldAnimateTodo(status, progress, running);
              return (
                <li key={item.id || item.content} className={`${status} ${animateTodo ? 'animating' : ''}`.trim()}>
                  <span className="todo-marker" aria-hidden="true">{statusIcon(status)}</span>
                  <span className={animateTodo ? 'assistant-processing-text' : ''}>{item.content}</span>
                  <em>{t.todoStatus?.[status] || status}</em>
                </li>
              );
            })}
          </ol>
        </div>
      )}
      {(progress.current || progress.items.length > 0) && (
        <div className="agent-activity">
          <strong>{t.activityTitle}</strong>
          {progress.current && (
            <div className={`agent-progress-current ${progress.current.status}`}>
              <span className="activity-pulse" aria-hidden="true" />
              <span className="assistant-processing-text">{progress.current.label}</span>
            </div>
          )}
          {progress.items.length > 0 && (
            <ol className="agent-progress-list">
              {progress.items.slice(-5).map((item, index) => (
                <li key={`${item.id}-${index}`}>
                  <Check size={13} />
                  <span>{item.label}</span>
                  {item.preview && (
                    <details className="tool-result-details">
                      <summary>
                        {t.toolResultPreview}
                        {item.previewTruncated ? ` ${t.toolResultTruncated}` : ''}
                      </summary>
                      <pre>{item.preview}</pre>
                    </details>
                  )}
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
      {progress.summary && <p>{progress.summary}</p>}
    </section>
  );
}

function shouldAnimateTodo(status, progress, running) {
  return status === 'in_progress' && Boolean(running && progress.current);
}

const Message = React.memo(function Message({ message, t, showProgress = true }) {
  if (message.kind === 'divider') {
    return <div className="feed-divider" role="separator"><span>{message.content}</span></div>;
  }
  const hasVoiceTtsPreparing = Boolean(message.role === 'assistant' && message.kind === 'message' && message.voiceTtsPreparing);
  const hasThinking = Boolean(
    message.role === 'assistant' &&
      message.kind === 'message' &&
      ((message.thinking && String(message.thinking).trim()) || message.thinkingRunning),
  );
  const hasInlineProgress = Boolean(
    message.role === 'assistant' &&
      showProgress &&
      !message.progressArchived &&
      (message.progress?.current ||
        (message.progress?.items?.length || 0) > 0 ||
        message.progress?.summary ||
        (message.todos?.length || 0) > 0),
  );
  const hasHeartbeat = Boolean(
    message.role === 'assistant' &&
      message.kind === 'message' &&
      message.heartbeatSec !== null &&
      message.heartbeatSec !== undefined &&
      !String(message.content || '').trim(),
  );
  if (
    message.role === 'assistant' &&
    message.kind === 'message' &&
    !String(message.content || '').trim() &&
    !hasVoiceTtsPreparing &&
    !hasThinking &&
    !hasInlineProgress &&
    !hasHeartbeat
  ) {
    return null;
  }
  if (message.kind === 'tool_result' && shouldHideToolResult(message.content)) {
    return null;
  }
  if (message.kind === 'agent_dialogue') {
    return (
      <article className="message assistant">
        <div className="bubble">
          <AgentDialogue message={message} />
        </div>
      </article>
    );
  }
  if (message.kind === 'diagram') {
    return (
      <article className="message assistant">
        <div className="bubble">
          <MermaidDiagram title={message.title} chart={message.content} />
        </div>
      </article>
    );
  }
  const icon = message.role === 'tool' ? <Terminal size={16} /> : <Bot size={16} />;
  const progressNode = hasInlineProgress ? (
    <AgentProgress
      progress={message.progress || emptyProgress()}
      todos={message.todos || []}
      t={t}
      running={Boolean(message.progressRunning)}
    />
  ) : null;
  const thinkingNode = hasThinking ? <ThinkingBlock content={message.thinking || ''} running={message.thinkingRunning} t={t} /> : null;
  const voiceTtsPreparingNode = hasVoiceTtsPreparing ? (
    <div className="voice-tts-preparing">
      <span className="activity-pulse" aria-hidden="true" />
      <span className="assistant-processing-text">{t.voiceTtsPreparing}</span>
    </div>
  ) : null;
  const heartbeatNode = hasHeartbeat ? (
    <div className="assistant-heartbeat">
      <span className="activity-pulse" aria-hidden="true" />
      <span>{format(t.heartbeatWaiting, { seconds: message.heartbeatSec })}</span>
    </div>
  ) : null;
  const usageNode = message.usage ? (
    <div className="message-usage">
      {format(t.usageLine, { input: message.usage.inputTokens ?? 0, output: message.usage.outputTokens ?? 0 })}
    </div>
  ) : null;
  const contentNode = <MarkdownContent content={message.content} />;
  const progressFirst = hasInlineProgress || message.contentPlacement === 'afterProgress';
  return (
    <article className={`message ${message.role}${message.voiceDraft ? ' voice-draft' : ''}`}>
      {message.role !== 'assistant' && <div className="avatar">{icon}</div>}
      <div className="bubble">
        {message.kind === 'tool_result' ? (
          <pre>{message.content}</pre>
        ) : message.role === 'assistant' ? (
          <>
            {thinkingNode}
            {voiceTtsPreparingNode}
            {heartbeatNode}
            {progressFirst && progressNode}
            {contentNode}
            {!progressFirst && progressNode}
            {usageNode}
          </>
        ) : (
          message.content
        )}
      </div>
    </article>
  );
});

function ApprovalToolInput({ toolName, toolInput, t }) {
  const input = toolInput && typeof toolInput === 'object' && !Array.isArray(toolInput) ? toolInput : {};
  const command = toolName === 'shell' ? String(input.command || '') : '';
  const diff = useMemo(() => {
    if (toolName === 'write_file') return buildApprovalDiff('', String(input.content || ''), t);
    if (toolName === 'edit_file') return buildApprovalDiff(String(input.old || ''), String(input.new || ''), t);
    return null;
  }, [toolName, input.content, input.old, input.new, t]);
  const rest = { ...input };
  if (command) delete rest.command;
  if (diff) {
    delete rest.content;
    delete rest.old;
    delete rest.new;
  }
  const restKeys = Object.keys(rest);
  return (
    <details open>
      <summary>{t.toolInput}</summary>
      {command && <pre className="approval-shell-command">{command}</pre>}
      {diff && (
        <div className="approval-diff" aria-label={t.toolInput}>
          {diff.map((row, index) => (
            <div key={`${row.kind}-${index}`} className={row.kind}>
              {row.text}
            </div>
          ))}
        </div>
      )}
      {(restKeys.length > 0 || (!command && !diff)) && <pre>{JSON.stringify(rest, null, 2)}</pre>}
    </details>
  );
}

const APPROVAL_DIFF_MAX_ROWS = 200;
const APPROVAL_DIFF_MAX_CHARS = 20000;

function buildApprovalDiff(oldText, newText, t) {
  // diffLines is O(n*m); on a large file it freezes the tab for seconds while the user is
  // waiting to approve. Report the sizes instead of computing a diff nobody can read.
  if (oldText.length > APPROVAL_DIFF_MAX_CHARS || newText.length > APPROVAL_DIFF_MAX_CHARS) {
    const template = t?.diffTooLarge || TRANSLATIONS.zh.diffTooLarge;
    return [{ kind: 'elided', text: format(template, { old: oldText.length, new: newText.length }) }];
  }
  const rows = [];
  let parts;
  try {
    parts = diffLines(oldText, newText);
  } catch {
    return null;
  }
  for (const part of parts) {
    const kind = part.added ? 'added' : part.removed ? 'removed' : 'context';
    const marker = part.added ? '+' : part.removed ? '-' : ' ';
    const lines = String(part.value || '').split('\n');
    if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
    for (const line of lines) {
      rows.push({ kind, text: `${marker} ${line}` });
      if (rows.length >= APPROVAL_DIFF_MAX_ROWS) {
        rows.push({ kind: 'elided', text: '…' });
        return rows;
      }
    }
  }
  return rows;
}

function AgentDialogue({ message }) {
  const participants = Array.isArray(message.participants) ? message.participants : [];
  const turns = Array.isArray(message.agentMessages) ? message.agentMessages : message.messages || [];
  const [selectedAgentId, setSelectedAgentId] = useState('all');
  const visibleTurns =
    selectedAgentId === 'all' ? turns : turns.filter((turn) => String(turn.agent_id || turn.agentId) === selectedAgentId);
  return (
    <section className="agent-dialogue">
      <div className="agent-dialogue-header">
        <div>
          <strong>{message.title || 'Agent 协作'}</strong>
          <span>{agentDialogueLabel(message.dialogueType)}</span>
        </div>
        {message.threadId && <code>{message.threadId}</code>}
      </div>
      {participants.length > 0 && (
        <div className="agent-perspective-tabs" aria-label="Agent perspective switcher">
          <button type="button" className={selectedAgentId === 'all' ? 'active' : ''} onClick={() => setSelectedAgentId('all')}>
            全部
          </button>
          {participants.map((participant) => {
            const id = String(participant.id || participant.agent_id || participant.name || '');
            if (!id) return null;
            return (
              <button
                type="button"
                key={id}
                className={selectedAgentId === id ? 'active' : ''}
                onClick={() => setSelectedAgentId(id)}
              >
                {participant.name || id}
              </button>
            );
          })}
        </div>
      )}
      <div className="agent-turns">
        {visibleTurns.map((turn, index) => (
          <div className="agent-turn" key={`${turn.agent_id || turn.agentId || 'agent'}-${turn.round || 0}-${index}`}>
            <div className="agent-turn-meta">
              <strong>{turn.agent_name || turn.agentName || turn.agent_id || 'Agent'}</strong>
              {turn.round ? <span>第 {turn.round} 轮</span> : null}
            </div>
            <div className="agent-turn-bubble">
              <MarkdownContent content={turn.content || ''} />
            </div>
          </div>
        ))}
        {!visibleTurns.length && <p className="agent-dialogue-empty">当前视角暂无发言。</p>}
      </div>
    </section>
  );
}

function agentDialogueLabel(type) {
  if (type === 'debate') return '辩论 transcript';
  if (type === 'parallel_delegate') return '并行子 Agent';
  return 'Agent transcript';
}

function ThinkingBlock({ content, running, t }) {
  const hasContent = String(content || '').trim();
  return (
    <section className={`thinking-block ${running ? 'running' : 'done'}`}>
      <div className="thinking-heading">
        <span className="thinking-dot" aria-hidden="true" />
        <strong>{running ? t.thinkingTitle : t.thinkingDoneTitle}</strong>
      </div>
      <div className={`thinking-content ${running ? 'assistant-processing-text' : ''}`}>
        {hasContent ? <MarkdownContent content={content} /> : <span>{t.thinkingPlaceholder}</span>}
      </div>
    </section>
  );
}

const MarkdownContent = React.memo(function MarkdownContent({ content }) {
  const normalizedContent = useMemo(() => normalizeMarkdownForDisplay(content || ''), [content]);
  const needsMath = MATH_MARKUP_PATTERN.test(normalizedContent);
  const needsHighlight = CODE_FENCE_PATTERN.test(normalizedContent);
  const [mathPlugins, setMathPlugins] = useState(mathPluginsCache);
  const [highlightPlugin, setHighlightPlugin] = useState(highlightPluginCache);

  useEffect(() => {
    if (!needsMath || mathPlugins) return undefined;
    let cancelled = false;
    loadMathPlugins()
      .then((plugins) => {
        if (!cancelled) setMathPlugins(plugins);
      })
      .catch(() => {
        // Math rendering stays plain text if the chunk cannot be fetched.
      });
    return () => {
      cancelled = true;
    };
  }, [needsMath, mathPlugins]);

  useEffect(() => {
    if (!needsHighlight || highlightPlugin) return undefined;
    let cancelled = false;
    loadHighlightPlugin()
      .then((plugin) => {
        if (!cancelled) setHighlightPlugin(plugin);
      })
      .catch(() => {
        // Code blocks stay unhighlighted if the chunk cannot be fetched.
      });
    return () => {
      cancelled = true;
    };
  }, [needsHighlight, highlightPlugin]);

  const useMath = needsMath && Boolean(mathPlugins);
  const useHighlight = needsHighlight && Boolean(highlightPlugin);
  const remarkPlugins = useMemo(
    () => (useMath ? [remarkGfm, mathPlugins.remarkMath] : BASE_REMARK_PLUGINS),
    [useMath, mathPlugins],
  );
  const rehypePlugins = useMemo(() => {
    if (!useMath && !useHighlight) return EMPTY_REHYPE_PLUGINS;
    const plugins = [];
    if (useMath) plugins.push(mathPlugins.rehypeKatex);
    if (useHighlight) plugins.push(highlightPlugin);
    return plugins;
  }, [useMath, useHighlight, mathPlugins, highlightPlugin]);

  if (!normalizedContent.trim()) return null;
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={{ pre: MermaidPre, table: MarkdownTable }}
      >
        {normalizedContent}
      </ReactMarkdown>
    </div>
  );
});

function MarkdownTable({ children, ...props }) {
  return (
    <div className="markdown-table-wrap">
      <table {...props}>{children}</table>
    </div>
  );
}

// navigator.clipboard rejects on http origins, when the document is not focused, and when
// the permission is denied; an unhandled rejection there left the user with no feedback.
async function copyTextToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    const template = activeTranslations?.copyFailed || 'Copy failed: {error}';
    emitToast({ message: format(template, { error: err?.message || String(err) }), tone: 'error' });
    return false;
  }
}

function MermaidPre({ children, ...props }) {
  const child = React.Children.toArray(children)[0];
  const className = child?.props?.className || '';
  if (/language-mermaid/.test(className)) {
    const chart = React.Children.toArray(child.props.children).join('');
    return <MermaidDiagram chart={chart} />;
  }
  const code = React.Children.toArray(child?.props?.children).join('').replace(/\n$/, '');
  const language = className.match(/language-([\w-]+)/)?.[1] || 'text';
  return (
    <div className="code-block">
      <div className="code-block-toolbar">
        <span>{language}</span>
        <button type="button" onClick={() => void copyTextToClipboard(code)} aria-label="Copy code" title="Copy code">
          <Copy size={14} />
        </button>
      </div>
      <pre {...props}>{children}</pre>
    </div>
  );
}

function MermaidDiagram({ chart, title }) {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState('');
  const [themeDark, setThemeDark] = useState(effectiveDarkTheme);
  const diagramId = useMemo(() => `mermaid-${crypto.randomUUID().replace(/-/g, '')}`, []);
  useEffect(() => subscribeThemeChange(setThemeDark), []);
  useEffect(() => {
    let cancelled = false;
    const source = normalizeMermaidForRender(chart);
    if (!source) {
      setSvg('');
      setError('');
      return;
    }
    const timer = window.setTimeout(() => {
      loadMermaid(themeDark)
        .then(async (instance) => {
          await instance.parse(source);
          return instance.render(diagramId, source);
        })
        .then((result) => {
          if (cancelled) return;
          setSvg(result.svg);
          setError('');
        })
        .catch((err) => {
          if (cancelled) return;
          setSvg('');
          setError(err?.message || String(err));
        });
    }, 300);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [chart, diagramId, themeDark]);

  return (
    <figure className="mermaid-card">
      {title && <figcaption>{title}</figcaption>}
      {svg ? (
        <div className="mermaid-canvas" dangerouslySetInnerHTML={{ __html: svg }} />
      ) : error ? (
        <details className="mermaid-error" open>
          <summary>图示渲染失败，显示 Mermaid 源码</summary>
          <pre>{chart}</pre>
        </details>
      ) : (
        <div className="mermaid-loading">正在渲染图示...</div>
      )}
    </figure>
  );
}

const MERMAID_SEQUENCE_RESERVED_IDS = new Set([
  'actor',
  'and',
  'alt',
  'break',
  'critical',
  'else',
  'end',
  'loop',
  'note',
  'opt',
  'par',
  'participant',
  'rect',
]);

function normalizeMermaidForRender(chart) {
  const source = String(chart || '').trim();
  if (!source.startsWith('sequenceDiagram')) return source;
  const reservedAliases = new Map();
  const lines = source.split('\n');
  const normalizedLines = lines.map((line) => {
    const declaration = line.match(/^(\s*(?:actor|participant)\s+)([A-Za-z_][\w-]*)(\s+as\b.*)$/i);
    if (!declaration) return line;
    const originalId = declaration[2];
    if (!MERMAID_SEQUENCE_RESERVED_IDS.has(originalId.toLowerCase())) return line;
    const safeId = `${originalId}_node`;
    reservedAliases.set(originalId, safeId);
    return `${declaration[1]}${safeId}${declaration[3]}`;
  });
  if (!reservedAliases.size) return normalizedLines.join('\n');
  return normalizedLines
    .map((line) => replaceMermaidSequenceReservedReferences(line, reservedAliases))
    .join('\n');
}

function replaceMermaidSequenceReservedReferences(line, aliases) {
  if (/^\s*(?:actor|participant)\s+/i.test(line)) return line;
  const colonIndex = line.indexOf(':');
  if (colonIndex < 0) return replaceMermaidIds(line, aliases);
  return `${replaceMermaidIds(line.slice(0, colonIndex), aliases)}${line.slice(colonIndex)}`;
}

function replaceMermaidIds(value, aliases) {
  let result = String(value || '');
  for (const [originalId, safeId] of aliases.entries()) {
    result = result.replace(new RegExp(`\\b${escapeRegExp(originalId)}\\b`, 'g'), safeId);
  }
  return result;
}

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function normalizeMarkdownForDisplay(content) {
  return normalizeCompressedMarkdownTables(normalizeMathMarkdown(content));
}

function normalizeMathMarkdown(content) {
  return String(content || '')
    .replace(/\\\[((?:.|\n)*?)\\\]/g, (_match, math) => `\n$$\n${math.trim()}\n$$\n`)
    .replace(/\\\((.+?)\\\)/g, (_match, math) => `$${math.trim()}$`);
}

// Every compressed-table candidate needs a line carrying at least four pipes. Scanning for
// that is O(n) and lets the whole (much more expensive) normalization be skipped for the
// overwhelming majority of messages, which carry no table at all.
// Native indexOf scans only: a per-character JS loop over a long message costs more than
// the normalization it is meant to avoid.
function hasCompressedTableCandidateLine(content) {
  let pipes = 0;
  let searchFrom = 0;
  let lineEnd = -1;
  for (;;) {
    const pipeAt = content.indexOf('|', searchFrom);
    if (pipeAt < 0) return false;
    if (pipeAt > lineEnd) {
      lineEnd = content.indexOf('\n', pipeAt);
      if (lineEnd < 0) lineEnd = content.length;
      pipes = 0;
    }
    pipes += 1;
    if (pipes >= 4) return true;
    searchFrom = pipeAt + 1;
  }
}

function normalizeCompressedMarkdownTables(content) {
  const source = String(content || '');
  if (!hasCompressedTableCandidateLine(source)) return source;
  const lines = normalizeMultilineCompressedMarkdownTables(source).split('\n');
  const normalizedLines = lines.flatMap((line) => {
    const splitRows = splitCompressedMarkdownTableLine(line);
    return splitRows.length ? splitRows : [line];
  });
  return normalizedLines.join('\n');
}

function normalizeMultilineCompressedMarkdownTables(content) {
  const lines = String(content || '').split('\n');
  const normalizedLines = [];
  let insideStandardTable = false;
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    // Rows of an already well-formed table pass through untouched: re-parsing them as a
    // compressed table infers the wrong column count and destroys the table.
    if (insideStandardTable) {
      if (typeof line === 'string' && line.includes('|')) {
        normalizedLines.push(line);
        continue;
      }
      insideStandardTable = false;
    }
    if (!isCompressedMarkdownTableStart(line)) {
      if (startsStandardMarkdownTable(lines, index)) insideStandardTable = true;
      normalizedLines.push(line);
      continue;
    }
    if (startsStandardMarkdownTable(lines, index)) {
      insideStandardTable = true;
      normalizedLines.push(line);
      continue;
    }

    const accumulator = createCompressedTableAccumulator(line);
    let cursor = index + 1;
    while (cursor < lines.length) {
      const table = accumulator.status();
      const nextLine = lines[cursor];
      const nextTrimmed = String(nextLine || '').trim();
      if (table?.isComplete && !nextTrimmed.startsWith('|')) break;
      if (!nextTrimmed && table) break;
      accumulator.append(nextLine);
      cursor += 1;
      if (accumulator.status()?.isComplete) {
        const nextAfterBlock = String(lines[cursor] || '').trim();
        if (!nextAfterBlock.startsWith('|')) break;
      }
    }

    const table = accumulator.table();
    if (!table) {
      normalizedLines.push(line);
      continue;
    }
    normalizedLines.push(...formatCompressedMarkdownTable(table));
    index = cursor - 1;
  }
  return normalizedLines.join('\n');
}

// Cells of a single markdown table row, with the optional outer pipes removed. Escaped
// pipes (\|) are cell content, not separators.
function markdownRowCells(line) {
  const trimmed = String(line ?? '').trim();
  if (!trimmed.includes('|')) return null;
  const cells = trimmed.split(/(?<!\\)\|/);
  if (cells.length && cells[0].trim() === '') cells.shift();
  if (cells.length && cells[cells.length - 1].trim() === '') cells.pop();
  return cells.length ? cells : null;
}

// A GFM delimiter row: every cell is one or more dashes with optional alignment colons
// (`| --- |`, `|---|`, `|:---|:-:|--:|`). Such a row always belongs to a standard table.
function isMarkdownTableDelimiterLine(line) {
  const cells = markdownRowCells(line);
  if (!cells || cells.length < 2) return false;
  return cells.every((cell) => /^\s*:?-+:?\s*$/.test(cell));
}

// A header row followed by a delimiter row of the same width is a standard table already.
function startsStandardMarkdownTable(lines, index) {
  const next = lines[index + 1];
  // Cheap reject first: this runs once per line of every rendered message.
  if (typeof next !== 'string' || !next.includes('|') || !next.includes('-')) return false;
  const headerCells = markdownRowCells(lines[index]);
  if (!headerCells || headerCells.length < 2) return false;
  if (isMarkdownTableDelimiterLine(lines[index])) return false;
  const delimiterCells = markdownRowCells(lines[index + 1]);
  if (!delimiterCells || delimiterCells.length !== headerCells.length) return false;
  return isMarkdownTableDelimiterLine(lines[index + 1]);
}

function isCompressedMarkdownTableStart(line) {
  const value = String(line || '');
  if ((value.match(/\|/g) || []).length < 4) return false;
  // A bare delimiter row is the second line of a standard table, never the start of a
  // compressed one; treating it as a start inferred 2 columns and mangled the table.
  if (isMarkdownTableDelimiterLine(value)) return false;
  return /\|\s*:?-{3,}:?\s*\|/.test(value);
}

function splitCompressedMarkdownTableLine(line) {
  const table = parseCompressedMarkdownTable(line);
  return table ? formatCompressedMarkdownTable(table) : [];
}

function parseCompressedMarkdownTable(line) {
  const value = String(line || '');
  if ((value.match(/\|/g) || []).length < 6) return null;
  if (isMarkdownTableDelimiterLine(value)) return null;
  const firstPipe = value.indexOf('|');
  if (firstPipe < 0) return null;
  const prefix = value.slice(0, firstPipe).trimEnd();
  const rawCells = value
    .slice(firstPipe)
    .split('|')
    .slice(1)
    .filter((cell) => cell.trim() !== '');
  return parseCompressedMarkdownCells(prefix, rawCells);
}

function createCompressedTableAccumulator(line) {
  const value = String(line || '');
  const firstPipe = value.indexOf('|');
  const prefix = firstPipe >= 0 ? value.slice(0, firstPipe).trimEnd() : '';
  const rawCells = [];
  let pending = '';
  // Layout (column count + separator run) never changes once it is fully visible,
  // so it is detected once and reused instead of re-parsing every cell per line.
  let layout = null;
  let layoutScanFrom = 1;

  function append(fragment, initial = false) {
    const source = `${pending}${initial ? '' : ' '}${initial ? String(fragment).slice(firstPipe + 1) : String(fragment)}`;
    const parts = source.split('|');
    pending = parts.pop() || '';
    rawCells.push(...parts.filter((cell) => cell.trim() !== ''));
  }

  function cellCount() {
    return rawCells.length + (pending.trim() ? 1 : 0);
  }

  function cellAt(index) {
    return index < rawCells.length ? rawCells[index] : index === rawCells.length && pending.trim() ? pending : undefined;
  }

  function detectLayout() {
    if (layout) return layout;
    const total = cellCount();
    for (let index = Math.max(1, layoutScanFrom); index < total; index += 1) {
      let separatorCount = 0;
      while (isMarkdownTableSeparatorCell(cellAt(index + separatorCount))) separatorCount += 1;
      if (separatorCount >= 1 && index >= 2) {
        // Only cache once the cell after the separator run has arrived, otherwise a
        // half-streamed run would freeze the wrong separator length.
        const settled = index + separatorCount < total;
        const candidate = { columnCount: index, separatorStart: index, separatorCellCount: separatorCount };
        if (settled) layout = candidate;
        return candidate;
      }
    }
    layoutScanFrom = Math.max(1, total - 1);
    return null;
  }

  function status() {
    const found = detectLayout();
    if (!found) return null;
    const total = cellCount();
    if (total < found.columnCount * 2) return null;
    const dataCount = total - found.separatorStart - found.separatorCellCount;
    return { isComplete: dataCount >= found.columnCount && dataCount % found.columnCount === 0 };
  }

  function table() {
    const found = detectLayout();
    if (!found) return null;
    const cells = pending.trim() ? [...rawCells, pending] : rawCells;
    if (cells.length < found.columnCount * 2) return null;
    const dataCount = cells.length - found.separatorStart - found.separatorCellCount;
    return {
      prefix,
      columnCount: found.columnCount,
      headerCells: cells.slice(0, found.columnCount),
      dataCells: cells.slice(found.separatorStart + found.separatorCellCount),
      isComplete: dataCount >= found.columnCount && dataCount % found.columnCount === 0,
    };
  }

  append(value, true);
  return { append, status, table };
}

function parseCompressedMarkdownCells(prefix, rawCells) {
  if (rawCells.length < 4) return null;
  // Nothing but delimiter cells means the caller handed over a standard table's delimiter
  // row; inferring a column count from it would fabricate a bogus 2-column table.
  if (rawCells.every((cell) => /^\s*:?-+:?\s*$/.test(cell))) return null;

  let columnCount = 0;
  let separatorCellCount = 0;
  let separatorStart = -1;
  for (let index = 1; index < rawCells.length; index += 1) {
    let separatorCount = 0;
    while (isMarkdownTableSeparatorCell(rawCells[index + separatorCount])) separatorCount += 1;
    if (separatorCount >= 1 && index >= 2) {
      columnCount = index;
      separatorCellCount = separatorCount;
      separatorStart = index;
      break;
    }
  }
  if (!columnCount || rawCells.length < columnCount * 2) return null;

  return {
    prefix,
    columnCount,
    headerCells: rawCells.slice(0, columnCount),
    dataCells: rawCells.slice(separatorStart + separatorCellCount),
    isComplete: (rawCells.length - separatorStart - separatorCellCount) >= columnCount
      && (rawCells.length - separatorStart - separatorCellCount) % columnCount === 0,
  };
}

function formatCompressedMarkdownTable(table) {
  const rows = [
    `| ${table.headerCells.map((cell) => normalizeTableCellText(cell)).join(' | ')} |`,
    `| ${Array.from({ length: table.columnCount }, () => '---').join(' | ')} |`,
  ];
  const completeDataCells = table.dataCells.slice(
    0,
    Math.floor(table.dataCells.length / table.columnCount) * table.columnCount,
  );
  for (let index = 0; index < completeDataCells.length; index += table.columnCount) {
    rows.push(`| ${completeDataCells
      .slice(index, index + table.columnCount)
      .map((cell) => normalizeTableCellText(cell))
      .join(' | ')} |`);
  }
  if (!rows.some((row) => isMarkdownTableSeparatorRow(row))) return [];
  return table.prefix ? [table.prefix, '', ...rows, ''] : ['', ...rows, ''];
}

function normalizeTableCellText(cell) {
  return String(cell || '').trim().replace(/\s+/g, ' ');
}

function isMarkdownTableSeparatorRow(row) {
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(String(row || '').trim());
}

function isMarkdownTableSeparatorCell(cell) {
  return /^\s*:?-{3,}:?\s*$/.test(String(cell || ''));
}

function isLikelyStreamingMarkdownTableRow(line) {
  const trimmed = String(line || '').trim();
  if (!trimmed.startsWith('|')) return false;
  if ((trimmed.match(/\|/g) || []).length < 2) return false;
  if (isMarkdownTableSeparatorRow(trimmed)) return true;
  return /\|\s*[^|\s][^|]*$/.test(trimmed);
}


// A 403 means this page is holding a token the server no longer accepts —
// normally because the server restarted while the tab stayed open. The token
// lives in the served HTML, so a reload is the whole fix. Reload once (guarded
// by sessionStorage so a genuinely rejected client cannot loop), and tell the
// user plainly if that did not help.
const STALE_TOKEN_RELOAD_KEY = 'langcode-stale-token-reload';

const STALE_TOKEN_FALLBACK_MESSAGE =
  'The server rejected this page (unauthorized). Reload the page; if it keeps failing, restart the server.';

function staleTokenMessage() {
  return activeTranslations?.staleTokenReloadFailed || STALE_TOKEN_FALLBACK_MESSAGE;
}

function markAuthorizedRequest() {
  try {
    window.sessionStorage.removeItem(STALE_TOKEN_RELOAD_KEY);
  } catch {
    // Nothing to clear when storage is unavailable.
  }
}

function handleUnauthorizedResponse(response) {
  if (!response || response.status !== 403) {
    if (response?.ok) markAuthorizedRequest();
    return false;
  }
  let alreadyReloaded = false;
  try {
    alreadyReloaded = window.sessionStorage.getItem(STALE_TOKEN_RELOAD_KEY) === '1';
  } catch {
    alreadyReloaded = false;
  }
  if (alreadyReloaded) {
    emitToast({ tone: 'error', fatal: true, message: staleTokenMessage() });
    return true;
  }
  try {
    window.sessionStorage.setItem(STALE_TOKEN_RELOAD_KEY, '1');
  } catch {
    // Private mode: reloading once is still the right move.
  }
  window.location.reload();
  return true;
}

async function api(path, body) {
  const options = body
    ? {
        method: 'POST',
        headers: { ...API_HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }
    : { headers: API_HEADERS };
  const response = await fetch(path, options);
  if (handleUnauthorizedResponse(response)) {
    return { ok: false, error: staleTokenMessage() };
  }
  return response.json();
}

createRoot(document.getElementById('root')).render(<App />);
