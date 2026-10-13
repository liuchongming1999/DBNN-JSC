"""Accessible hover/click presentation shared by the original-code layer guide."""

LAYER_CSS = r'''
.layer-guide{margin-top:24px}.layer-guide-heading{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
.layer-guide-heading h3{margin:0}.layer-guide-heading>span{font-size:12px;color:var(--muted)}
.layer-table{min-width:750px;table-layout:fixed;font-size:13px}.layer-table thead th:nth-child(1){width:17%}.layer-table thead th:nth-child(2){width:28%}.layer-table thead th:nth-child(3){width:27%}.layer-table thead th:nth-child(4){width:28%}
.layer-table tbody th{background:#f7fafb;padding:0;font-weight:400}.layer-table td{padding:14px 13px}.layer-table tbody tr:hover td{background:#f8fcfb}
.layer-table td>strong{font-size:20px;color:var(--accent)}.layer-table .muted{font-size:12px}
.layer-trigger{display:flex;flex-direction:column;gap:4px;align-items:flex-start;width:100%;background:transparent;color:var(--ink);border-radius:0;padding:13px 12px;text-align:left;min-height:112px}
.layer-trigger strong{font-size:15px}.layer-index{font:11px/1.2 ui-monospace,monospace;color:var(--accent)}.layer-trigger-hint{font-size:11px;color:var(--accent);font-weight:400}
.layer-trigger:hover,.layer-trigger[aria-expanded=true]{background:#e5f2ee}.layer-trigger:focus-visible,.layer-close:focus-visible,.layer-popover-body:focus-visible{outline:3px solid #c68a25;outline-offset:-3px}
.layer-badge{display:block;width:max-content;max-width:100%;font-size:10px;line-height:1.6;color:#426070;background:#edf3f5;border-radius:4px;padding:2px 6px;margin-top:6px;font-weight:400}
.layer-protocol{font-size:12px;color:var(--muted);background:#f3f6f7;padding:12px 14px;border-radius:6px}
.layer-full{margin:12px 0 0;border:1px solid var(--line);border-radius:8px;padding:0}.layer-full>summary{padding:12px 14px;color:var(--accent);font-size:13px}
.layer-full article{padding:18px 20px;border-top:1px solid var(--line);break-inside:auto}.layer-full article>h3{margin:0 0 12px;font-size:18px}.layer-full .layer-detail-content{max-width:940px}
.layer-detail-content h4,.layer-popover-body h4{font-size:13px;color:var(--accent);margin:18px 0 6px}.layer-detail-content h4:first-child,.layer-popover-body h4:first-child{margin-top:0}
.layer-detail-content p,.layer-popover-body p{margin:7px 0}.layer-detail-content ol,.layer-popover-body ol{padding-left:22px;margin:6px 0}.layer-detail-content li,.layer-popover-body li{margin:6px 0}
.layer-detail-content,.layer-popover-body{font-size:13px;line-height:1.75}.layer-math{padding:10px 12px;background:#edf6f3;border-radius:6px;font-size:13px}
.layer-source{overflow-wrap:anywhere;background:#f3f6f7;border-radius:6px;padding:10px 12px;font:11px/1.7 ui-monospace,monospace;color:var(--muted)}
.layer-popover{position:fixed;z-index:1000;width:590px;max-width:calc(100vw - 24px);background:white;border:1px solid #91b8ac;border-radius:12px;box-shadow:0 16px 55px #163f4f33;overflow:hidden}
.layer-popover[hidden]{display:none}.layer-popover-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:13px 18px;border-bottom:1px solid var(--line);background:#edf6f3}
.layer-popover-head h3{margin:2px 0 0;font-size:20px}.layer-popover-mode{font-size:11px;color:var(--muted)}.layer-close{background:transparent;color:#355564;font-size:26px;line-height:1;padding:8px 10px;flex:none}
.layer-popover-body{padding:16px 18px;overflow:auto;overscroll-behavior:contain;max-height:min(65vh,650px)}.layer-popover table{font-size:11.5px}.layer-popover td,.layer-popover th{padding:7px 8px}
@media(max-width:700px){.layer-popover{left:12px!important;right:12px;top:12px!important;width:auto;max-width:none}.layer-popover-body{max-height:calc(100dvh - 110px)}.layer-full article{padding:14px}.layer-guide-heading>span{font-size:11px}}
@media print{.layer-popover{display:none!important}.layer-table{min-width:0;font-size:11px}.layer-trigger{display:flex!important;padding:8px;min-height:0}.layer-trigger-hint,.layer-index,#layer-help,.layer-guide-heading>span{display:none}.layer-full{border:0}.layer-full>summary{display:none}.layer-full article{padding:12px 0}.layer-full h4{break-after:avoid}.layer-table tr{break-inside:avoid}}
'''


LAYER_JS = r'''
(() => {
  const panel = document.getElementById('layer-popover');
  const body = document.getElementById('layer-popover-body');
  const title = document.getElementById('layer-popover-title');
  const mode = document.getElementById('layer-popover-mode');
  const buttons = [...document.querySelectorAll('.layer-trigger')];
  const closeButton = panel.querySelector('.layer-close');
  const full = document.querySelector('.layer-full');
  let active = null, pinned = false, timer = null, suppressFocus = false;
  function cancelHide() { clearTimeout(timer); }
  function position() {
    if (!active || panel.hidden) return;
    const rect = active.getBoundingClientRect(), w = panel.offsetWidth, h = panel.offsetHeight;
    const pad = 12, gap = 12;
    const left = Math.max(pad, Math.min(innerWidth - w - pad, rect.right + gap));
    const top = Math.max(pad, Math.min(innerHeight - h - pad, rect.top));
    panel.style.left = left + 'px'; panel.style.top = top + 'px';
  }
  function show(button, pin = false) {
    cancelHide();
    if (pinned && !pin) return;
    if (active !== button) {
      if (active) active.setAttribute('aria-expanded', 'false');
      const detail = document.getElementById('layer-detail-' + button.dataset.layer);
      title.textContent = detail.dataset.title;
      body.innerHTML = detail.querySelector('.layer-detail-content').innerHTML;
      body.scrollTop = 0;
    }
    active = button; pinned = pin;
    button.setAttribute('aria-expanded', 'true'); panel.hidden = false;
    mode.textContent = pin ? '已固定 · 可滚动阅读 · Esc 关闭' : '悬停预览 · 点击名称固定';
    position();
  }
  function hide(restore = false) {
    cancelHide();
    const previous = active;
    if (previous) previous.setAttribute('aria-expanded', 'false');
    active = null; pinned = false; panel.hidden = true;
    if (restore && previous) {
      suppressFocus = true; previous.focus({preventScroll: true}); suppressFocus = false;
    }
  }
  function scheduleHide() {
    cancelHide();
    timer = setTimeout(() => {
      if (!pinned && !panel.matches(':hover') && !panel.contains(document.activeElement)
          && active && !active.matches(':hover') && document.activeElement !== active) hide();
    }, 220);
  }
  buttons.forEach(button => {
    button.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') show(button); });
    button.addEventListener('pointerleave', scheduleHide);
    button.addEventListener('focus', () => { if (!suppressFocus && button.matches(':focus-visible')) show(button); });
    button.addEventListener('blur', scheduleHide);
    button.addEventListener('click', () => {
      if (active === button && pinned) { hide(); return; }
      show(button, true); body.focus({preventScroll: true});
    });
    button.addEventListener('keydown', event => {
      if (event.key === 'ArrowDown' && active === button) { event.preventDefault(); show(button, true); body.focus(); }
    });
  });
  panel.addEventListener('pointerenter', cancelHide);
  panel.addEventListener('pointerleave', scheduleHide);
  panel.addEventListener('focusout', scheduleHide);
  closeButton.addEventListener('click', () => hide(true));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && active) { event.preventDefault(); hide(true); }
  });
  document.addEventListener('pointerdown', event => {
    if (active && !panel.contains(event.target) && !event.target.closest('.layer-trigger')) hide();
  });
  addEventListener('resize', position);
  addEventListener('scroll', () => { if (pinned) position(); else if (active) hide(); }, {passive:true});
  let wasOpen = false;
  addEventListener('beforeprint', () => { wasOpen = full.open; full.open = true; hide(); });
  addEventListener('afterprint', () => { full.open = wasOpen; });
})();
'''
