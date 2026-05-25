import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import {
  ArrowDown,
  Bot,
  ChevronDown,
  ChevronRight,
  Check,
  CircleAlert,
  Code2,
  Folder,
  AudioLines,
  MoreHorizontal,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Send,
  Settings,
  SlidersHorizontal,
  Square,
  Terminal,
  X,
} from 'lucide-react';
import './styles.css';
import 'katex/dist/katex.min.css';

const MERMAID_CONFIG = {
  startOnLoad: false,
  securityLevel: 'strict',
  theme: 'base',
  themeVariables: {
    fontFamily: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    primaryColor: '#f7f7f7',
    primaryBorderColor: '#d1d1d1',
    primaryTextColor: '#0d0d0d',
    lineColor: '#8f8f8f',
    secondaryColor: '#eef8f5',
    tertiaryColor: '#ffffff',
  },
};
let mermaidModulePromise = null;

function loadMermaid() {
  if (!mermaidModulePromise) {
    mermaidModulePromise = import('mermaid').then((module) => {
      const instance = module.default;
      instance.initialize(MERMAID_CONFIG);
      return instance;
    });
  }
  return mermaidModulePromise;
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

function normalizeTtsVoiceOptions(voices = []) {
  const normalized = Array.isArray(voices) ? voices.filter((voice) => voice?.id) : [];
  const output = normalized.map((voice) =>
    voice.id === DEFAULT_TTS_VOICE_OPTION.id ? { ...DEFAULT_TTS_VOICE_OPTION, ...voice, previewUrl: voice.previewUrl || DEFAULT_TTS_VOICE_OPTION.previewUrl, previewReady: true } : voice,
  );
  const hasDefault = output.some((voice) => voice.id === DEFAULT_TTS_VOICE_OPTION.id);
  return hasDefault ? output : [DEFAULT_TTS_VOICE_OPTION, ...output];
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
    openDirectoryPicker: 'Choose workspace',
    nativePickerPrompt: 'Choose workspace directory',
    directoryPickerTitle: 'Choose workspace directory',
    parentDirectory: 'Parent',
    homeDirectory: 'Home',
    currentDirectory: 'Current',
    selectedDirectory: 'Selected directory',
    useThisDirectory: 'Use selected directory',
    close: 'Close',
    loadingDirectories: 'Loading directories...',
    noDirectories: 'No child directories',
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
    openDirectoryPicker: '选择工作目录',
    nativePickerPrompt: '选择工作目录',
    directoryPickerTitle: '选择工作目录',
    parentDirectory: '上级目录',
    homeDirectory: '用户目录',
    currentDirectory: '当前目录',
    selectedDirectory: '已选择目录',
    useThisDirectory: '使用所选目录',
    close: '关闭',
    loadingDirectories: '正在加载目录...',
    noDirectories: '没有子目录',
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

function getInitialLanguage() {
  return localStorage.getItem('langcode-language') || 'zh';
}

function getInitialSidebarCollapsed() {
  return localStorage.getItem('langcode-sidebar-collapsed') === 'true';
}

const ACTIVE_SESSION_STORAGE_KEY = 'langcode-active-session-id';

function getStoredActiveSessionId() {
  return localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY) || '';
}

function getInitialTtsVoiceId() {
  return localStorage.getItem('langcode-tts-voice-id') || 'default';
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
  const [agentProgress, setAgentProgress] = useState({ items: [], current: null, summary: '' });
  const [todos, setTodos] = useState([]);
  const [error, setError] = useState('');
  const [approvalEdit, setApprovalEdit] = useState('');
  const [workspaceInput, setWorkspaceInput] = useState('');
  const [selectedWorkspace, setSelectedWorkspace] = useState('');
  const [directoryPickerOpen, setDirectoryPickerOpen] = useState(false);
  const [directoryData, setDirectoryData] = useState(null);
  const [directoryLoading, setDirectoryLoading] = useState(false);
  const [directoryError, setDirectoryError] = useState('');
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
  const ttsSpeechSequenceRef = useRef({});
  const ttsDelayedDisplayRef = useRef({});
  const ttsBargeInTokenRef = useRef('');
  const ttsSuppressedTokensRef = useRef(new Set());
  const markdownDeltaBuffersRef = useRef({});
  const ttsVoiceIdRef = useRef(getInitialTtsVoiceId());
  const voiceActiveRef = useRef(false);
  const voiceConversationActiveRef = useRef(false);
  const voiceRestartTimerRef = useRef(null);
  const bargeInTriggeredRef = useRef(false);
  const bargeInContextRef = useRef(null);

  const activeSessionBusy = Boolean(sessionId && runningSessions[sessionId]);
  const interactionBusy = busy || activeSessionBusy;
  const anySessionBusy = Object.keys(runningSessions).length > 0;
  const workspaceBusy = busy || anySessionBusy;
  const visibleSessions = mergeSessions(sessionId, sessions, draftSessions);
  const ttsVoiceOptions = normalizeTtsVoiceOptions(
    status?.tts?.voices?.length
      ? status.tts.voices
      : [
          { id: 'xuefen', name: '雪芬', style: '自定义音色', builtIn: true },
          { id: 'wangju', name: '汪菊', style: '自定义音色', builtIn: true },
        ],
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
    localStorage.setItem('langcode-tts-voice-id', ttsVoiceIdRef.current);
  }, [ttsVoiceId]);

  useEffect(() => {
    if (!ttsVoiceOptions.length) return;
    if (!ttsVoiceOptions.some((voice) => voice.id === ttsVoiceId)) {
      setTtsVoiceId(ttsVoiceOptions[0].id);
    }
  }, [ttsVoiceOptions, ttsVoiceId]);

  useEffect(() => {
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
    };
  }, []);

  useEffect(() => {
    activeSessionRef.current = sessionId;
    if (sessionId) {
      localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, sessionId);
      writeActiveSessionLocation(sessionId);
    } else {
      localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
      writeActiveSessionLocation('');
    }
  }, [sessionId]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    localStorage.setItem('langcode-language', language);
  }, [language]);

  useEffect(() => {
    localStorage.setItem('langcode-sidebar-collapsed', String(sidebarCollapsed));
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
          localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
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
    { clearInput = false, reuseUserMessageId = '', ignoreStaleRunning = false, voiceInterrupt = null } = {},
  ) {
    const text = String(rawText || '').trim();
    const activeSessionId = sessionId;
    if (!activeSessionId) {
      setError(t.chooseWorkspaceFirst);
      return;
    }
    if (!text || busy || activeRunsRef.current[activeSessionId] || (!ignoreStaleRunning && runningSessions[activeSessionId])) return;
    stickToBottomRef.current = true;
    setShowJumpToLatest(false);
    stopTtsPlayback({ clearQueue: true, stopVoice: !voiceConversationActiveRef.current, suppressCurrent: true });
    if (clearInput) setInput('');
    setSlashMenuOpen(false);
    setError('');
    setAgentProgress({ items: [], current: null, summary: '' });
    setTodos([]);
    const assistantId = crypto.randomUUID();
    const runId = crypto.randomUUID();
    const visibleMessages = messagesRef.current;
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
    activeRunsRef.current[activeSessionId] = { runId, assistantId, userMessageId, controller: null, initialMessages };
    sessionLiveMessagesRef.current[activeSessionId] = initialMessages;
    if (activeSessionRef.current === activeSessionId) {
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
    if (shouldDiscardInterruptedRun(messagesRef.current, activeRun)) {
      const nextMessages = messagesRef.current.filter(
        (message) => message.id !== activeRun.assistantId && message.id !== activeRun.userMessageId,
      );
      messagesRef.current = nextMessages;
      sessionLiveMessagesRef.current[activeSessionId] = nextMessages;
      setMessages(nextMessages);
      setAgentProgress({ items: [], current: null, summary: '' });
      setTodos([]);
      clearRunIfCurrent(activeSessionId, activeRun.runId);
      refreshSessions();
      return;
    }
    setAgentProgress((current) => finishAgentProgress(current, t));
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
    setAgentProgress((current) => finishAgentProgress(current, t));
    setTodos([]);
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
    for (const token of tokensToSuppress) ttsSuppressedTokensRef.current.add(token);
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
        }
      } else {
        ttsIncomingTextRef.current = {};
        ttsQueuedSpeechRef.current = {};
        ttsPlayedSpeechRef.current = {};
        ttsSpeechSequenceRef.current = {};
        ttsDelayedDisplayRef.current = {};
        markdownDeltaBuffersRef.current = {};
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
    if (
      !ttsEnabled ||
      !voiceConversationActiveRef.current ||
      !targetSessionId ||
      !assistantId ||
      activeSessionRef.current !== targetSessionId
    ) return;
    if (pendingApproval || sessionPendingApprovalsRef.current[targetSessionId]) return;
    const token = ttsToken(targetSessionId, assistantId);
    if (ttsSuppressedTokensRef.current.has(token)) return;
    if (ttsPlaybackTokenRef.current && ttsPlaybackTokenRef.current !== token) {
      stopTtsPlayback({ clearQueue: true, stopVoice: true, suppressCurrent: true });
    }
    if (!ttsPlaybackTokenRef.current) ttsPlaybackTokenRef.current = token;
    refreshTtsBargeInContext(targetSessionId, assistantId);
    const selectedVoiceId = ttsVoiceIdRef.current || 'default';
    const useCustomVoice = selectedVoiceId !== 'default';
    const shouldDelayDisplay = Boolean(delayDisplay && useCustomVoice);
    if (shouldDelayDisplay && !ttsDelayedDisplayRef.current[token]) {
      ttsDelayedDisplayRef.current[token] = { released: false };
    }
    const incomingDelta = delta ? takeNovelTtsIncomingDelta(token, delta) : '';
    const current = `${ttsChunkBuffersRef.current[token] || ''}${incomingDelta}`;
    const { chunks, remainder } = splitTtsText(current, { force, compact: useCustomVoice });
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
      ttsQueueRef.current.push({
        type: 'speech',
        speechId: nextTtsSpeechId(token),
        token,
        sessionId: targetSessionId,
        assistantId,
        voiceId: selectedVoiceId,
        userText: previousUser?.content || '',
        text,
        normalizedText,
        displayText: shouldDelayDisplay ? chunk : '',
        delayDisplay: shouldDelayDisplay,
        queuedAt: Date.now(),
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

  function shouldDelayCustomVoiceDisplay(targetSessionId) {
    return Boolean(
      ttsEnabled &&
        voiceConversationActiveRef.current &&
        activeSessionRef.current === targetSessionId &&
        (ttsVoiceIdRef.current || 'default') !== 'default',
    );
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

  function clearMarkdownDeltaBuffer(targetSessionId, assistantId) {
    delete markdownDeltaBuffersRef.current[markdownDeltaToken(targetSessionId, assistantId)];
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
        while (inFlight.length < ttsPrefetchWindow()) {
          const nextItem = takeNextTtsQueueItem({ speechOnly: true });
          if (!nextItem) break;
          inFlight.push({
            item: nextItem,
            promise: prepareTtsStreamItem(nextItem),
            prepared: null,
          });
        }

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
          if (!isCurrentTtsSpeechItem(next.item)) {
            releasePreparedTts(prepared);
            continue;
          }
          if (prepared.blobs.length === 0) {
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
      ttsPumpActiveRef.current = false;
      if (ttsQueueRef.current.length > 0) void pumpTtsQueue();
    }
  }

  async function prepareTtsEntry(entry) {
    if (!entry.prepared) entry.prepared = await entry.promise;
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

  function ttsPrefetchWindow() {
    const voiceId = ttsVoiceIdRef.current || 'default';
    return voiceId && voiceId !== 'default' ? 3 : 1;
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

  async function prepareTtsStreamItem(item) {
    ensureTtsBargeInListening(item);
    const controller = new AbortController();
    const timeoutMs = ttsSynthesisTimeoutMs(item.text, item.voiceId);
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
    ttsCurrentAbortControllersRef.current.add(controller);
    const blobs = [];
    try {
      const response = await fetch('/api/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: item.text, voiceId: item.voiceId || 'default' }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const detail = await response.text().catch(() => '');
        if (activeSessionRef.current === item.sessionId) {
          setVoiceError(format(t.voiceUnavailable, { error: detail || `${response.status} ${response.statusText}` }));
        }
        return { item, blobs };
      }
      if (!response.body) {
        const blob = await response.blob();
        blobs.push(blob);
        return { item, blobs };
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
          const blob = decodeTtsStreamLine(line, item);
          if (blob) blobs.push(blob);
        }
      }
      if (buffer.trim()) {
        const blob = decodeTtsStreamLine(buffer, item);
        if (blob) blobs.push(blob);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        if (activeSessionRef.current === item.sessionId) {
          setVoiceError(format(t.voiceUnavailable, { error: err.message || t.requestFailed }));
        }
      }
    } finally {
      window.clearTimeout(timeoutId);
      ttsCurrentAbortControllersRef.current.delete(controller);
    }
    return { item, blobs, audioItems: blobs.map(createPreparedTtsAudio) };
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
    if (payload.type !== 'audio' || !payload.audio) return null;
    return base64AudioBlob(payload.audio, payload.contentType || 'audio/wav');
  }

  async function playPreparedTtsItem(prepared) {
    if (!claimPreparedTtsPlayback(prepared.item)) {
      releasePreparedTts(prepared);
      return;
    }
    if (prepared.item.delayDisplay) {
      appendDelayedAssistantContent(prepared.item.sessionId, prepared.item.assistantId, prepared.item.displayText || prepared.item.text);
    }
    const audioItems = prepared.audioItems?.length ? prepared.audioItems : prepared.blobs.map(createPreparedTtsAudio);
    for (const audioItem of audioItems) {
      if (!isCurrentTtsSpeechItem(prepared.item)) return;
      await playPreparedTtsAudio(audioItem, prepared.item);
    }
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
    await new Promise((resolve) => {
      const { audio, url } = audioItem;
      ttsObjectUrlRef.current = url;
      ttsAudioRef.current = audio;
      ttsPlayingRef.current = true;
      const finish = () => {
        if (ttsAudioResolveRef.current === finish) ttsAudioResolveRef.current = null;
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
    for (const audioItem of prepared?.audioItems || []) {
      releasePreparedTtsAudio(audioItem);
    }
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
    ttsQueueRef.current = ttsQueueRef.current.filter((item) => item.token !== token);
    delete ttsChunkBuffersRef.current[token];
    delete ttsIncomingTextRef.current[token];
    delete ttsQueuedSpeechRef.current[token];
    delete ttsPlayedSpeechRef.current[token];
    delete ttsSpeechSequenceRef.current[token];
    delete ttsDelayedDisplayRef.current[token];
    if (ttsPlaybackTokenRef.current === token) {
      ttsPlaybackTokenRef.current = '';
      ttsBargeInTokenRef.current = '';
    }
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
      const response = await fetch(`${voice.previewUrl}?t=${Date.now()}`);
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
    setAgentProgress(makeRunningProgress);
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
        if (activeSessionRef.current === approvalSessionId) setAgentProgress((current) => finishAgentProgress(current, t));
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
        setAgentProgress((current) => ({ ...current, current: null }));
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

  async function loadDirectories(path) {
    setDirectoryLoading(true);
    setDirectoryError('');
    const suffix = path ? `?path=${encodeURIComponent(path)}` : '';
    const response = await api(`/api/directories${suffix}`);
    setDirectoryLoading(false);
    if (!response.ok) {
      setDirectoryError(response.error || t.requestFailed);
      return;
    }
    setDirectoryData(response);
    setSelectedWorkspace(response.path || selectedWorkspace);
  }

  async function applyWorkspace(workspacePath = selectedWorkspace) {
    const nextWorkspace = workspacePath.trim();
    if (!nextWorkspace || workspaceBusy) return;
    setBusy(true);
    setError('');
    const response = await api('/api/workspace', { workspace: nextWorkspace });
    setBusy(false);
    if (!response.ok) {
      setError(response.error || t.requestFailed);
      return;
    }
    setStatus(response);
    setWorkspaceInput(response.workspace || nextWorkspace);
    setSelectedWorkspace(response.workspace || nextWorkspace);
    setDirectoryPickerOpen(false);
    const nextSessionId = `web-${Date.now().toString(36)}`;
    activeSessionRef.current = nextSessionId;
    setSessionId(nextSessionId);
    setMessages([]);
    setAgentProgress({ items: [], current: null, summary: '' });
    setPendingApproval(null);
    setPendingApprovalAssistantId('');
    setApprovalProcessing(false);
    refreshSessions();
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
      setTodos(response.todos);
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, runId }),
        signal: controller.signal,
      });
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
      if (shouldDelayCustomVoiceDisplay(streamSessionId)) {
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
      if (isActiveSession) setAgentProgress((current) => updateAgentProgress(current, event, t));
      updateSessionMessages(streamSessionId, (current) =>
        updateAssistantProgressInMessages(current, assistantId, (progress) => updateAgentProgress(progress, event, t)),
      );
      return;
    }
    if (event.type === 'delta') {
      if (shouldDelayCustomVoiceDisplay(streamSessionId)) {
        setAssistantVoiceTtsPreparing(streamSessionId, responseAssistantId, true);
        queueAssistantTtsDelta(streamSessionId, responseAssistantId, event.content || '', { delayDisplay: true });
        return;
      }
      const displayDelta = takeMarkdownDisplayDelta(streamSessionId, responseAssistantId, event.content || '');
      updateSessionMessages(streamSessionId, (current) =>
        current.map((message) =>
          message.id === responseAssistantId
            ? {
                ...message,
                content: `${message.content}${displayDelta}`,
                thinkingRunning: false,
                contentPlacement: contentPlacementForMessage({
                  ...message,
                  content: `${message.content}${displayDelta}`,
                }),
              }
            : message,
        ),
      );
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
      if (isActiveSession) {
        setTodos(nextTodos);
        setAgentProgress((current) => ({ ...current, summary: event.summary || current.summary }));
      }
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
      const flushedMarkdown = flushMarkdownDisplayDelta(streamSessionId, responseAssistantId);
      if (flushedMarkdown) {
        updateSessionMessages(streamSessionId, (current) =>
          appendAssistantContentInMessages(current, responseAssistantId, flushedMarkdown),
        );
      }
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
      const delayCustomVoiceDisplay = !event.cancelled && isActiveSession && shouldDelayCustomVoiceDisplay(streamSessionId);
      const flushedMarkdown = event.cancelled || delayCustomVoiceDisplay
        ? ''
        : flushMarkdownDisplayDelta(streamSessionId, responseAssistantId);
      if (flushedMarkdown) {
        updateSessionMessages(streamSessionId, (current) =>
          appendAssistantContentInMessages(current, responseAssistantId, flushedMarkdown),
        );
      }
      if (isActiveSession) setAgentProgress((current) => finishAgentProgress(current, t));
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
      if (!event.cancelled && isActiveSession && voiceConversationActiveRef.current && !ttsEnabled) {
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
      if (!isActiveSession) refreshSessions();
      return;
    }
    if (event.type === 'error') {
      clearMarkdownDeltaBuffer(streamSessionId, responseAssistantId);
      stopVoiceConversation();
      if (isActiveSession) {
        setError(event.error || t.requestFailed);
        setAgentProgress((current) => ({ ...current, current: null }));
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
      if (!isActiveSession) refreshSessions();
    }
  }

  function voiceWsUrl() {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${scheme}//${window.location.host}/api/asr/stream`;
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
            if (bargeIn && !bargeInTriggeredRef.current && isBargeInIntent(partialText)) {
              bargeInTriggeredRef.current = true;
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
          const finalText = String(payload.text || input).trim();
          const semanticState = payload.semanticVad?.state || '';
          clearVoiceAutoFinishTimer();
          voiceSocketRef.current = null;
          submitVoiceCaptureText(finalText, { bargeIn, semanticState });
          return;
        }
        if (payload.type === 'error') {
          const message = payload.error || t.requestFailed;
          setVoiceError(message);
          setError(format(t.voiceUnavailable, { error: message }));
          void stopVoiceCapture({ cancel: true, submit: false });
        }
      };

      ws.onerror = () => {
        setVoiceError(t.requestFailed);
        setError(format(t.voiceUnavailable, { error: t.requestFailed }));
        void stopVoiceCapture({ cancel: true, submit: false });
      };

      await startVoiceAudioPipeline(ws);
    } catch (err) {
      setVoiceError(err.message);
      setError(err.name === 'NotAllowedError' ? t.voicePermissionDenied : format(t.voiceUnavailable, { error: err.message }));
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

  function clearVoiceAutoFinishTimer() {
    if (voiceAutoFinishTimerRef.current) {
      window.clearTimeout(voiceAutoFinishTimerRef.current);
    }
    voiceAutoFinishTimerRef.current = null;
    voiceAutoFinishTextRef.current = '';
    voiceAutoFinishModeRef.current = '';
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
    voiceAutoFinishTimerRef.current = window.setTimeout(() => {
      if (!voiceActiveRef.current || voiceStoppingRef.current || voiceSocketRef.current !== ws) return;
      if (voiceLastPartialTextRef.current !== text) return;
      if (Date.now() - changedAt < targetStableMs - 100) return;
      setVoiceStatus(t.voiceFinalizing);
      voiceAutoFinishTimerRef.current = null;
      voiceAutoFinishTextRef.current = '';
      voiceAutoFinishModeRef.current = '';
      voiceStoppingRef.current = true;
      voiceSocketRef.current = null;
      try {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'cancel' }));
        ws.close();
      } catch {
        // Best effort: the transcript is already stable enough to submit.
      }
      submitVoiceCaptureText(text, { bargeIn, semanticState });
    }, delay);
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
        autoGainControl: true,
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
      const chunkSamples = 8000;
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
    const activeRun = activeRunsRef.current[id];
    const cachedMessages = activeRun ? sessionLiveMessagesRef.current[id] || activeRun.initialMessages || [] : [];
    if (activeRun && cachedMessages.length > 0) {
      const sessionMeta = visibleSessions.find((session) => session.id === id);
      const cachedApproval = sessionPendingApprovalsRef.current[id] || null;
      activeSessionRef.current = id;
      setSessionId(id);
      setMessages(cachedMessages);
      setPendingApproval(cachedApproval);
      setPendingApprovalAssistantId(cachedApproval ? activeRun.assistantId || findLastAssistantId(cachedMessages) : '');
      setApprovalEdit(cachedApproval ? JSON.stringify(cachedApproval.toolInput, null, 2) : '');
      setApprovalProcessing(false);
      setAgentProgress({ items: [], current: null, summary: '' });
      setTodos([]);
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
    setMessages([]);
    setPendingApproval(null);
    setPendingApprovalAssistantId('');
    setApprovalProcessing(false);
    setAgentProgress({ items: [], current: null, summary: '' });
    setTodos([]);
    setError('');
    setInput('');
    setSlashMenuOpen(false);
    setSessionMenu(null);
    setRenamingSessionId('');
    const response = await api(`/api/session?sessionId=${encodeURIComponent(id)}`);
    if (activeSessionRef.current !== id) return;
    if (!response.ok) {
      setError(response.error || t.requestFailed);
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
    setMessages(restoredMessages);
    setTodos(loadedTodos);
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
        setAgentProgress({ items: [], current: null, summary: '' });
        setTodos([]);
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
      setAgentProgress({ items: [], current: null, summary: '' });
      setTodos([]);
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
      <aside className="sidebar" aria-label="Session navigation" aria-hidden={sidebarCollapsed}>
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

        <section className="message-feed" ref={scrollRef} onScroll={handleMessageFeedScroll}>
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

        {error && (
          <div className="error-strip">
            <CircleAlert size={16} />
            {error}
          </div>
        )}
        {voiceError && (
          <div className="error-strip">
            <CircleAlert size={16} />
            {voiceError}
          </div>
        )}

        <div className="composer-stack">
          {showJumpToLatest && (
            <button type="button" className="jump-latest" onClick={() => scrollToLatest()} aria-label={t.jumpToLatest}>
              <ArrowDown size={15} />
              {t.jumpToLatest}
            </button>
          )}

          {pendingApproval && (
            <section className="approval-strip" aria-label="Tool approval">
              <div className="approval-summary">
                <SlidersHorizontal size={16} />
                <div>
                  <strong>{format(t.allowTool, { tool: pendingApproval.toolName })}</strong>
                  <span>{pendingApproval.payload?.risk?.reason || t.reviewToolInput}</span>
                </div>
              </div>
              <details>
                <summary>{t.toolInput}</summary>
                <pre>{JSON.stringify(pendingApproval.toolInput, null, 2)}</pre>
              </details>
              <textarea
                value={approvalEdit}
                onChange={(event) => setApprovalEdit(event.target.value)}
                rows={3}
                aria-label={t.approvalEditLabel}
              />
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
                <button onClick={() => approve('edit')}>{t.edit}</button>
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

function archiveAssistantProgress(messages) {
  return messages.map((message) => {
    if (message.role !== 'assistant' || !hasAssistantProgress(message)) return message;
    return {
      ...message,
      progress: emptyProgress(),
      todos: [],
      progressRunning: false,
      progressArchived: true,
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

function splitTtsText(text, { force = false, compact = false } = {}) {
  let buffer = String(text || '').trimStart();
  const chunks = [];
  const minReadyLength = compact ? 72 : 48;
  while (buffer.length >= minReadyLength) {
    const boundary = findBalancedTtsBoundary(buffer, { compact });
    if (boundary <= 0) break;
    chunks.push(buffer.slice(0, boundary).trim());
    buffer = buffer.slice(boundary).trimStart();
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

function findBalancedTtsBoundary(text, { compact = false } = {}) {
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
    marks: ['。', '！', '？', '!', '?', '；', ';', '\n'],
  });
  if (strong > 0) return strong;
  if (length >= target) {
    const weak = nearestTtsCut(text, {
      from: Math.max(min, target - 10),
      target,
      to: endLimit,
      marks: ['，', ',', '、', '：', ':'],
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

function isBargeInIntent(text) {
  const compact = String(text || '').replace(/\s+/g, '');
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
  if (String(message.content || '').trim()) return 'beforeProgress';
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

function Message({ message, t, showProgress = true }) {
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
  if (
    message.role === 'assistant' &&
    message.kind === 'message' &&
    !String(message.content || '').trim() &&
    !hasVoiceTtsPreparing &&
    !hasThinking &&
    !hasInlineProgress
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
  const contentNode = <MarkdownContent content={message.content} />;
  const progressFirst = message.contentPlacement === 'afterProgress';
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
            {progressFirst && progressNode}
            {contentNode}
            {!progressFirst && progressNode}
          </>
        ) : (
          message.content
        )}
      </div>
    </article>
  );
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

function MarkdownContent({ content }) {
  const normalizedContent = normalizeMathMarkdown(content || '');
  if (!normalizedContent.trim()) return null;
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{ pre: MermaidPre }}
      >
        {normalizedContent}
      </ReactMarkdown>
    </div>
  );
}

function MermaidPre({ children, ...props }) {
  const child = React.Children.toArray(children)[0];
  const className = child?.props?.className || '';
  if (/language-mermaid/.test(className)) {
    const chart = React.Children.toArray(child.props.children).join('');
    return <MermaidDiagram chart={chart} />;
  }
  return <pre {...props}>{children}</pre>;
}

function MermaidDiagram({ chart, title }) {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState('');
  const diagramId = useMemo(() => `mermaid-${crypto.randomUUID().replace(/-/g, '')}`, []);
  useEffect(() => {
    let cancelled = false;
    const source = String(chart || '').trim();
    if (!source) {
      setSvg('');
      setError('');
      return;
    }
    loadMermaid()
      .then((instance) => instance.render(diagramId, source))
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
    return () => {
      cancelled = true;
    };
  }, [chart, diagramId]);

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

function normalizeMathMarkdown(content) {
  return content
    .replace(/\\\[((?:.|\n)*?)\\\]/g, (_match, math) => `\n$$\n${math.trim()}\n$$\n`)
    .replace(/\\\((.+?)\\\)/g, (_match, math) => `$${math.trim()}$`);
}

function isLikelyStreamingMarkdownTableRow(line) {
  const trimmed = String(line || '').trim();
  if (!trimmed.startsWith('|')) return false;
  if ((trimmed.match(/\|/g) || []).length < 2) return false;
  if (/^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(trimmed)) return true;
  return /\|\s*[^|\s][^|]*$/.test(trimmed);
}

function DirectoryPicker({ data, error, loading, selectedPath, t, onClose, onLoad, onSelect, onUse, busy }) {
  const roots = data?.roots || [];
  const directories = data?.directories || [];
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={t.directoryPickerTitle}>
      <section className="directory-dialog">
        <header className="directory-header">
          <div>
            <strong>{t.directoryPickerTitle}</strong>
            <span>{data?.path || t.loadingDirectories}</span>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label={t.close}>
            <X size={17} />
          </button>
        </header>

        <div className="directory-jumpbar">
          {data?.parent && (
            <button type="button" onClick={() => onLoad(data.parent)}>
              {t.parentDirectory}
            </button>
          )}
          {roots.map((root) => (
            <button type="button" key={root.path} onClick={() => onLoad(root.path)}>
              {root.path === data?.home ? t.homeDirectory : root.path === data?.workspace ? t.currentDirectory : root.name}
            </button>
          ))}
        </div>

        {error && (
          <div className="directory-error">
            <CircleAlert size={15} />
            {error}
          </div>
        )}

        <div className="directory-list">
          {loading && <div className="directory-empty">{t.loadingDirectories}</div>}
          {!loading && directories.length === 0 && <div className="directory-empty">{t.noDirectories}</div>}
          {!loading &&
            directories.map((directory) => (
              <button
                type="button"
                key={directory.path}
                className={directory.path === selectedPath ? 'selected' : ''}
                onClick={() => onSelect(directory.path)}
                onDoubleClick={() => onLoad(directory.path)}
              >
                <Folder size={16} />
                <span>{directory.name}</span>
              </button>
            ))}
        </div>

        <footer className="directory-footer">
          <label>
            <span>{t.selectedDirectory}</span>
            <input value={selectedPath || ''} readOnly />
          </label>
          <button type="button" onClick={() => onUse(selectedPath)} disabled={busy || !selectedPath}>
            {t.useThisDirectory}
          </button>
        </footer>
      </section>
    </div>
  );
}

async function api(path, body) {
  const options = body
    ? {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }
    : undefined;
  const response = await fetch(path, options);
  return response.json();
}

createRoot(document.getElementById('root')).render(<App />);
