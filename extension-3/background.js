const RULE_ID_GOOGLE = 1001;
const RULE_ID_BING = 1002;

async function enableSafeSearchRules() {
	try {
		const rulesToAdd = [
			{
				id: RULE_ID_GOOGLE,
				priority: 1,
				action: {
					type: "redirect",
					redirect: {
						transform: {
							queryTransform: {
								addOrReplaceParams: [
									{ key: "safe", value: "active" }
								]
							}
						}
					}
				},
				condition: {
					resourceTypes: ["main_frame", "sub_frame"],
					urlFilter: "||google."
				}
			},
			{
				id: RULE_ID_BING,
				priority: 1,
				action: {
					type: "redirect",
					redirect: {
						transform: {
							queryTransform: {
								addOrReplaceParams: [
									{ key: "adlt", value: "strict" }
								]
							}
						}
					}
				},
				condition: {
					resourceTypes: ["main_frame", "sub_frame"],
					urlFilter: "||bing.com"
				}
			}
		];

		await chrome.declarativeNetRequest.updateDynamicRules({
			addRules: rulesToAdd,
			removeRuleIds: []
		});
		return { success: true };
	} catch (error) {
		console.error('[background] Failed to enable SafeSearch rules:', error);
		return { success: false, error: error.message || 'SafeSearch is locked by browser settings' };
	}
}

async function disableSafeSearchRules() {
	try {
		await chrome.declarativeNetRequest.updateDynamicRules({
			addRules: [],
			removeRuleIds: [RULE_ID_GOOGLE, RULE_ID_BING]
		});
		return { success: true };
	} catch (error) {
		console.error('[background] Failed to disable SafeSearch rules:', error);
		return { success: false, error: error.message || 'SafeSearch is locked by browser settings' };
	}
}

async function setMinorMode(isMinor) {
	await chrome.storage.local.set({ isMinor });
	if (isMinor) {
		return await enableSafeSearchRules();
	} else {
		return await disableSafeSearchRules();
	}
}

chrome.runtime.onInstalled.addListener(async () => {
	const { isMinor } = await chrome.storage.local.get("isMinor");
	if (isMinor) {
		const result = await enableSafeSearchRules();
		if (!result.success) {
			console.warn('[background] Could not enable SafeSearch on install:', result.error);
		}
	}
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
	(async () => {
		if (message && message.type === "SET_SAFE") {
			const result = await setMinorMode(!!message.minor);
			sendResponse({ ok: result.success, error: result.error });
		}
	})();
	return true;
});
