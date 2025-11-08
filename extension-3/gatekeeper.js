(function(){
	'use strict';

	const OVERLAY_ID = '__face_age_gate_overlay__';
	const BLOCK_BODY_CLASS = '__face_age_gate_block__';
	const ONE_MINUTE = 60 * 1000;

	function now(){ return Date.now(); }

	function makeOverlay(){
		let overlay = document.getElementById(OVERLAY_ID);
		if (overlay) return overlay;
		overlay = document.createElement('div');
		overlay.id = OVERLAY_ID;
		overlay.style.position = 'fixed';
		overlay.style.top = '0';
		overlay.style.left = '0';
		overlay.style.width = '100vw';
		overlay.style.height = '100vh';
		overlay.style.zIndex = '2147483647';
		overlay.style.background = 'linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(2, 6, 23, 0.98))';
		overlay.style.backdropFilter = 'blur(6px)';
		overlay.style.display = 'flex';
		overlay.style.alignItems = 'center';
		overlay.style.justifyContent = 'center';
		overlay.style.color = '#e2e8f0';
		overlay.style.fontFamily = '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Ubuntu, Cantarell, sans-serif';

		const card = document.createElement('div');
		card.style.maxWidth = '520px';
		card.style.margin = '20px';
		card.style.padding = '28px';
		card.style.borderRadius = '16px';
		card.style.border = '1px solid rgba(148, 163, 184, 0.2)';
		card.style.background = 'rgba(15, 23, 42, 0.85)';
		card.style.boxShadow = '0 20px 60px rgba(0,0,0,0.4)';
		card.style.textAlign = 'center';

		const title = document.createElement('h2');
		title.textContent = 'Verification Required';
		title.style.margin = '0 0 10px 0';
		title.style.fontSize = '22px';
		title.style.fontWeight = '800';
		title.style.background = 'linear-gradient(135deg, #0ea5e9, #a855f7)';
		title.style.webkitBackgroundClip = 'text';
		title.style.webkitTextFillColor = 'transparent';

		const desc = document.createElement('p');
		desc.textContent = 'To continue browsing, please complete a quick age verification.';
		desc.style.margin = '8px 0 18px 0';
		desc.style.color = '#94a3b8';

		const btn = document.createElement('button');
		btn.textContent = 'Start verification';
		btn.style.padding = '12px 18px';
		btn.style.border = 'none';
		btn.style.borderRadius = '10px';
		btn.style.fontWeight = '700';
		btn.style.cursor = 'pointer';
		btn.style.background = 'linear-gradient(135deg, #0ea5e9, #0284c7)';
		btn.style.color = '#fff';
		btn.addEventListener('click', async () => {
			try {
				await chrome.runtime.sendMessage({ type: 'OPEN_VERIFICATION' });
			} catch (_) {}
		});

		card.appendChild(title);
		card.appendChild(desc);
		card.appendChild(btn);
		overlay.appendChild(card);
		return overlay;
	}

	function blockPage(){
		if (!document.getElementById(OVERLAY_ID)){
			const overlay = makeOverlay();
			document.documentElement.appendChild(overlay);
			document.documentElement.classList.add(BLOCK_BODY_CLASS);
			document.documentElement.style.overflow = 'hidden';
		}
	}

	function unblockPage(){
		const overlay = document.getElementById(OVERLAY_ID);
		if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay);
		document.documentElement.classList.remove(BLOCK_BODY_CLASS);
		document.documentElement.style.overflow = '';
	}

	async function checkAndApply(){
		try{
			const { verificationAllowedUntil } = await chrome.storage.local.get('verificationAllowedUntil');
			if (!verificationAllowedUntil || typeof verificationAllowedUntil !== 'number' || now() > verificationAllowedUntil){
				blockPage();
			}else{
				unblockPage();
			}
		}catch(_){ /* ignore */ }
	}

	// initial check
	checkAndApply();

	// react to storage changes
	try {
		chrome.storage.onChanged.addListener((changes, area) => {
			if (area === 'local' && changes.verificationAllowedUntil){
				checkAndApply();
			}
		});
	} catch(_) {}

	// periodic re-check in case some pages load after script init
	setInterval(checkAndApply, ONE_MINUTE);
})();



