// VitalsAI Dynamic Chatbot Override
(function(){
  const PAGE_MAP = {
    '/heart':'heart','/brain':'brain','/diabetes':'diabetes','/kidney':'kidney',
    '/eye':'eye','/lung':'lung','/bmi':'bmi','/history':'history','/':'home'
  };
  const pathName = location.pathname.replace(/\/$/,'') || '/';
  const page = PAGE_MAP[pathName] || (document.title || 'VitalsAI').toLowerCase();

  const KNOWLEDGE = {
    heart:{doctor:'Cardiologist', danger:['chest pain','jaw or left arm pain','heavy sweating','shortness of breath','fainting'], tips:['Check blood pressure and cholesterol regularly','Reduce salt, fried foods, and smoking','Walk daily if your doctor says it is safe','Manage stress with sleep, breathing exercises, or yoga']},
    brain:{doctor:'Neurologist', danger:['face drooping','arm weakness','speech difficulty','sudden severe headache','vision loss'], tips:['Treat FAST symptoms as an emergency','Control blood pressure, sugar, and cholesterol','Avoid smoking','Maintain regular walking and a consistent sleep routine']},
    diabetes:{doctor:'Diabetologist / Endocrinologist', danger:['very high sugar','confusion','vomiting','dehydration','blurred vision with weakness'], tips:['Avoid sugary drinks and refined carbohydrates','Track glucose and HbA1c regularly','Use fiber, protein, and portion control','Walk 30 to 45 minutes daily if medically safe']},
    kidney:{doctor:'Nephrologist', danger:['very low urine output','face or leg swelling','blood in urine','very high blood pressure','breathing difficulty'], tips:['Control blood pressure and blood sugar','Track creatinine and eGFR reports','Avoid overuse of painkillers','Follow your doctor’s sodium and fluid advice']},
    eye:{doctor:'Ophthalmologist', danger:['sudden vision loss','severe eye pain','eye injury','flashes or floaters','red eye with headache'], tips:['Follow the 20-20-20 rule for screen breaks','Get regular eye checkups if you have diabetes','Do not use medicated eye drops without medical advice','Use UV protection and maintain eye hygiene']},
    lung:{doctor:'Pulmonologist / Oncologist', danger:['coughing blood','severe breathlessness','chest pain','blue lips','unexplained weight loss'], tips:['Stop smoking and avoid secondhand smoke','Reduce dust and pollution exposure','Track breathing symptoms','Maintain follow-up scans and reports']},
    bmi:{doctor:'Dietitian / Physician', danger:['rapid unexplained weight loss','severe weakness','breathlessness','swelling'], tips:['A BMI of 18.5 to 24.9 is generally considered normal','Set calorie targets based on your goal','Prioritize protein, fiber, and hydration','Avoid crash diets']}
  };

  const GENERAL_HEALTH = {
    fever:{keys:['fever','temperature'], title:'Fever', info:['Rest and drink enough fluids','Monitor your temperature','Use paracetamol only if it is safe for you','See a doctor for fever lasting more than 3 days, rash, breathing difficulty, or confusion']},
    cold:{keys:['cold','cough','sore throat'], title:'Cold / Cough', info:['Drink warm fluids','Steam inhalation may help congestion','Rest and use a mask around others','See a doctor for blood in cough, breathlessness, or high fever']},
    headache:{keys:['headache','migraine'], title:'Headache', info:['Drink water and check your sleep','Take screen breaks','Seek urgent care for sudden severe headache, weakness, vomiting, or vision changes']},
    acidity:{keys:['acidity','gas','heartburn','acid reflux'], title:'Acidity / Gas', info:['Reduce spicy and oily food','Avoid heavy late-night meals','Eat smaller frequent meals','Chest pain with sweating or breathlessness needs emergency care']},
    cholesterol:{keys:['cholesterol','ldl','hdl','triglyceride'], title:'Cholesterol', info:['Reduce fried foods and trans fats','Add fiber, oats, nuts, and vegetables','Regular walking can help','Medication decisions should be based on your lipid report and doctor’s advice']},
    anemia:{keys:['anemia','haemoglobin','hemoglobin','hb low'], title:'Anemia / Low Hemoglobin', info:['Eat iron-rich foods such as leafy greens, beans, and dates','Vitamin C helps iron absorption','See a doctor for fatigue, breathlessness, or very low hemoglobin']},
    thyroid:{keys:['thyroid','tsh','t3','t4'], title:'Thyroid', info:['Diagnosis is based on TSH, T3, and T4 reports','Take prescribed medicine at the same time daily','See a physician or endocrinologist for weight change, palpitations, or fatigue']},
    stress:{keys:['stress','anxiety','tension'], title:'Stress / Anxiety', info:['Try slow breathing for 5 minutes','Maintain a sleep routine and take screen breaks','Daily walking or yoga can help','Seek immediate help for self-harm thoughts']},
    water:{keys:['water','hydration','dehydration'], title:'Hydration', info:['Pale yellow urine is usually a good hydration sign','You may need more fluids in heat or during exercise','Heart and kidney patients should follow doctor-specific fluid advice']},
    vaccine:{keys:['vaccine','vaccination'], title:'Vaccination', info:['Vaccine schedules depend on age and medical history','Ask your doctor about flu, COVID, hepatitis, tetanus, and other relevant vaccines','If you missed a vaccine, ask a physician about catch-up dosing']}
  };

  function shouldUseEnglish(){
    return true;
  }

  function currentDisease(){
    if (KNOWLEDGE[page]) return page;
    const text = (document.body.innerText || '').toLowerCase();
    return ['heart','brain','diabetes','kidney','eye','lung','bmi'].find(k => text.includes(k)) || 'heart';
  }

  function formEntries(){
    const entries = [];
    document.querySelectorAll('input,select,textarea').forEach(el => {
      if (!el.id || el.id === 'chatInput' || el.type === 'hidden' || el.type === 'button' || el.type === 'file') return;
      const raw = el.value;
      if (raw === '' || raw == null) return;
      const label = el.closest('.fg,div,label')?.querySelector?.('label')?.innerText?.trim() || el.getAttribute('placeholder') || el.id;
      const value = el.tagName === 'SELECT' ? (el.options[el.selectedIndex]?.text || raw) : raw;
      entries.push({label:label, value:value, raw:raw, text:(label + ' ' + value).toLowerCase(), num:Number(raw)});
    });
    return entries;
  }

  function formSnapshot(){
    return formEntries().map(e => e.label + ': ' + e.value);
  }

  function resultSnapshot(){
    const boxes = [...document.querySelectorAll('#result,.result,.result-card')].filter(el => {
      const style = getComputedStyle(el);
      return style.display !== 'none' && el.innerText.trim().length > 0;
    });
    return boxes[0]?.innerText.trim().replace(/\n{3,}/g,'\n') || '';
  }

  function localRiskHint(){
    const data = formSnapshot().join(' ').toLowerCase();
    let score = 0;
    const nums = [...data.matchAll(/(-?\d+(\.\d+)?)/g)].map(m => Number(m[1]));
    nums.forEach(n => {
      if (n > 180) score += 12;
      else if (n > 140) score += 8;
      else if (n > 100) score += 4;
      if (n > 30 && n < 80) score += 4;
    });
    if (/yes|smoker|current|poor|sedentary|present|high/i.test(data)) score += 18;
    return Math.max(5, Math.min(95, score + 20));
  }

  function clampScore(n){
    return Math.max(3, Math.min(97, Math.round(n)));
  }

  function fieldNum(entries, patterns){
    for (const e of entries) {
      if (patterns.some(p => p.test(e.label.toLowerCase())) && !Number.isNaN(e.num)) return e.num;
    }
    return null;
  }

  function fieldHas(entries, patterns, yesWords){
    return entries.some(e => patterns.some(p => p.test(e.label.toLowerCase())) && yesWords.some(w => e.value.toLowerCase().includes(w)));
  }

  function predictFromForm(){
    const d = currentDisease();
    const entries = formEntries();
    const reasons = [];
    const protect = [];
    let score = 12;

    function add(points, text){ score += points; reasons.push(text); }
    function good(points, text){ score -= points; protect.push(text); }

    const age = fieldNum(entries, [/age|ઉંમર/]);
    const bmi = fieldNum(entries, [/bmi|body mass/]);
    const bp = fieldNum(entries, [/bp|blood pressure|systolic|pressure/]);
    const glucose = fieldNum(entries, [/glucose|sugar|rbs|fbs/]);
    const chol = fieldNum(entries, [/cholesterol|lipid/]);

    if (age != null && age > 55) add(14, 'Age above 55 is a risk factor');
    else if (age != null && age < 40) good(5, 'Age is relatively lower');
    if (bmi != null && bmi >= 30) add(12, 'BMI is in the obese range');
    else if (bmi != null && bmi >= 18.5 && bmi < 25) good(8, 'BMI is in the normal range');
    if (bp != null && bp >= 140) add(14, 'Blood pressure is high');
    if (glucose != null && glucose >= 126) add(16, 'Glucose is in the high range');
    if (chol != null && chol >= 220) add(12, 'Cholesterol is high');

    if (d === 'heart') {
      if (fieldHas(entries, [/smok|tobacco/], ['yes','smoker'])) add(14, 'Smoking increases heart risk');
      if (fieldHas(entries, [/diabetes/], ['yes','pre'])) add(10, 'Diabetes is linked with higher heart risk');
      if (fieldHas(entries, [/chest|pain/], ['yes'])) add(20, 'Chest pain is an important symptom');
      if (fieldHas(entries, [/breath|shortness/], ['yes'])) add(16, 'Shortness of breath is a warning symptom');
      if (fieldHas(entries, [/activity|exercise/], ['yes','regular','active'])) good(8, 'Regular activity is protective');
    } else if (d === 'diabetes') {
      const insulin = fieldNum(entries, [/insulin/]);
      const family = fieldNum(entries, [/pedigree|family/]);
      if (insulin != null && insulin > 160) add(8, 'Insulin is elevated');
      if (family != null && family > .7) add(8, 'Family history score is high');
      if (fieldHas(entries, [/thirst|urination|urine/], ['yes'])) add(14, 'Thirst or frequent urination can be important diabetes symptoms');
      if (fieldHas(entries, [/activity|exercise/], ['regular','active'])) good(8, 'Regular activity helps glucose control');
    } else if (d === 'kidney') {
      const creat = fieldNum(entries, [/creatinine/]);
      const egfr = fieldNum(entries, [/egfr|glomerular/]);
      const hb = fieldNum(entries, [/hemoglobin|haemoglobin|hb/]);
      const albumin = fieldNum(entries, [/albumin/]);
      if (creat != null && creat > 1.3) add(22, 'Serum creatinine is high');
      if (egfr != null && egfr < 60) add(24, 'eGFR below 60 suggests higher kidney risk');
      if (hb != null && hb < 12) add(8, 'Hemoglobin is low');
      if (albumin != null && albumin > 1) add(12, 'Urine albumin is elevated');
      if (fieldHas(entries, [/edema|swelling|pedal/], ['yes','present'])) add(12, 'Swelling or edema may be a kidney-related symptom');
    } else if (d === 'brain') {
      if (fieldHas(entries, [/hypertension/], ['yes'])) add(18, 'Hypertension increases stroke risk');
      if (fieldHas(entries, [/heart disease/], ['yes'])) add(14, 'Heart disease is linked with higher stroke risk');
      if (fieldHas(entries, [/smok/], ['smokes','formerly','yes'])) add(10, 'Smoking is a stroke risk factor');
      if (fieldHas(entries, [/weakness|speech/], ['yes'])) add(24, 'Weakness or speech difficulty is an emergency warning sign');
    } else if (d === 'lung') {
      if (fieldHas(entries, [/smoking/], ['current','former','passive'])) add(20, 'Smoking exposure is a major lung risk factor');
      if (fieldHas(entries, [/stage/], ['iii','iv'])) add(24, 'Advanced stage suggests higher risk');
      if (fieldHas(entries, [/family/], ['yes'])) add(8, 'Family history is a risk factor');
      if (fieldHas(entries, [/breath|cough/], ['yes'])) add(14, 'Breathing or cough symptoms are important');
    } else if (d === 'eye') {
      if (fieldHas(entries, [/pain|blur|vision|red/], ['yes'])) add(20, 'Eye symptoms are present');
      if (glucose != null && glucose >= 126) add(10, 'High glucose is linked with eye complications');
    }

    if (!entries.length) {
      return {disease:d, score:null, status:'Need data', reasons:['Please fill the form values before prediction'], protect:[]};
    }

    score = clampScore(score);
    const status = score >= 60 ? 'At Risk' : score >= 35 ? 'Moderate Risk' : 'Safe / Low Risk';
    if (!reasons.length) reasons.push('No major high-risk value was detected');
    return {disease:d, score:score, status:status, reasons:reasons.slice(0,5), protect:protect.slice(0,4)};
  }

  function predictionHtml(gu){
    const p = predictFromForm();
    if (p.score == null) {
      return '<b>I need data for prediction.</b><br>Fill age, vital signs, lifestyle, and symptoms, then ask me to predict.';
    }
    const color = p.score >= 60 ? '#c53030' : p.score >= 35 ? '#b45309' : '#276749';
    return '<b>Chatbot Rough Prediction</b>' +
      '<br>Disease/Page: <b>' + p.disease + '</b>' +
      '<br>Status: <b style="color:' + color + '">' + p.status + '</b>' +
      '<br>Probability hint: <b>' + p.score + '%</b>' +
      '<br><br><b>Explanation:</b><br>• ' + p.reasons.join('<br>• ') +
      (p.protect.length ? '<br><br><b>Protective points:</b><br>• ' + p.protect.join('<br>• ') : '') +
      '<br><br><small>Note: This is a chatbot screening estimate. Use the medical model result and a doctor’s advice for final decisions.</small>';
  }

  function generalHealthHtml(topic, gu){
    return '<b>' + topic.title + '</b><br>• ' + topic.info.join('<br>• ') + '<br><br><small>Do not delay care for emergency symptoms.</small>';
  }

  function answer(question){
    const q = question.toLowerCase();
    const gu = shouldUseEnglish();
    const d = currentDisease();
    const k = KNOWLEDGE[d] || KNOWLEDGE.heart;
    const form = formSnapshot();
    const result = resultSnapshot();
    const score = localRiskHint();

    if (/predict|prediction|estimate|calculate risk|check risk|risk level|risk score|probability|screening/.test(q)) {
      return predictionHtml(gu);
    }

    if (/form|input|fill|value|data|field|fields/.test(q)) {
      return '<b>Form help:</b><br>Fill personal data, vital signs, lifestyle, and symptoms. Do not leave required fields empty.<br><br><b>Current values:</b><br>' + (form.length ? form.join('<br>') : 'No values entered yet.');
    }

    if (/result|score|probability|risk|safe|at risk|status/.test(q)) {
      if (result) return '<b>Current result:</b><br>' + result.replace(/\n/g,'<br>') + '<br><br>This is screening support, not a final diagnosis.';
      return 'No result yet. Fill the form and click <b>Predict Now</b>. From current entered values, the rough risk hint is about <b>' + score + '%</b>.';
    }

    if (/doctor|specialist|consult|physician|appointment/.test(q)) {
      return 'For ' + d + ', you usually consult a <b>' + k.doctor + '</b>.';
    }

    if (/danger|emergency|urgent|symptom|sign|serious|warning|pain|breathing/.test(q)) {
      return '<b>' + d + ' warning signs:</b><br>• ' + k.danger.join('<br>• ') + '<br><br>If any symptom is sudden or severe, seek urgent medical care.';
    }

    if (/tip|advice|lifestyle|diet|exercise|food|sleep|habit|routine/.test(q)) {
      return '<b>' + d + ' lifestyle tips:</b><br>• ' + k.tips.join('<br>• ');
    }

    if (/bmi|calorie|weight|height|nutrition|body mass/.test(q)) {
      return 'For BMI, enter weight and height. For calories, enter age, gender, activity, and goal. Normal BMI is generally 18.5 to 24.9.';
    }

    for (const topic of Object.values(GENERAL_HEALTH)) {
      if (topic.keys.some(key => q.includes(key))) return generalHealthHtml(topic, gu);
    }

    if (/hi|hello|hey|greetings/.test(q)) {
      return 'Hi! I am the dynamic VitalsAI assistant. Ask me about ' + d + ' form fields, symptoms, results, lifestyle tips, or doctors.';
    }

    return 'I can predict and explain from the current <b>' + d + '</b> form, and I can also answer general health questions.<br>Try: "predict my risk", "explain result", "what should I do for fever?", "high BP?", or "which doctor?".';
  }

  function applyAdvancedChatUi(){
    if (document.getElementById('vitalsAdvancedChatUi')) return;
    const style = document.createElement('style');
    style.id = 'vitalsAdvancedChatUi';
    style.textContent = '#chatBubble{position:fixed!important;right:22px!important;bottom:22px!important;z-index:99999!important;display:flex!important;flex-direction:column!important;align-items:flex-end!important;gap:12px!important;font-family:Inter,DM Sans,Segoe UI,Arial,sans-serif!important}#chatToggle{width:62px!important;height:62px!important;border-radius:22px!important;border:1px solid rgba(255,255,255,.22)!important;background:linear-gradient(135deg,#2563eb 0%,#0f766e 52%,#111827 100%)!important;color:#fff!important;cursor:pointer!important;font-size:0!important;box-shadow:0 18px 42px rgba(37,99,235,.35),0 6px 18px rgba(15,23,42,.18)!important;position:relative!important;transition:transform .18s ease,box-shadow .18s ease!important}#chatToggle:before{content:"AI";font-size:18px;font-weight:900;letter-spacing:.3px}#chatToggle:after{content:"";position:absolute;right:9px;top:9px;width:11px;height:11px;border-radius:50%;background:#22c55e;border:2px solid #fff;box-shadow:0 0 0 4px rgba(34,197,94,.18)}#chatToggle:hover{transform:translateY(-2px) scale(1.03)!important;box-shadow:0 22px 50px rgba(37,99,235,.42),0 8px 22px rgba(15,23,42,.2)!important}#chatBox{width:min(420px,calc(100vw - 28px))!important;height:min(680px,calc(100vh - 116px))!important;background:rgba(255,255,255,.96)!important;border:1px solid rgba(148,163,184,.35)!important;border-radius:24px!important;box-shadow:0 28px 80px rgba(15,23,42,.28)!important;overflow:hidden!important;display:none!important;flex-direction:column!important;backdrop-filter:blur(18px)!important}#chatBox.open{display:flex!important;animation:vitalsChatIn .18s ease-out}#chatHeader{background:radial-gradient(circle at 20% 0%,rgba(255,255,255,.22),transparent 34%),linear-gradient(135deg,#1d4ed8 0%,#0f766e 52%,#172554 100%)!important;color:#fff!important;padding:18px 18px!important;display:flex!important;align-items:center!important;gap:13px!important;position:relative!important}#chatHeader:after{content:"AI + form-aware health assistant";position:absolute;left:70px;bottom:9px;font-size:10px;font-weight:700;opacity:.72;letter-spacing:.2px}#chatHeader .avatar{width:42px!important;height:42px!important;border-radius:16px!important;background:rgba(255,255,255,.18)!important;border:1px solid rgba(255,255,255,.22)!important;display:flex!important;align-items:center!important;justify-content:center!important;font-size:21px!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.22)!important}#chatHeader .info{min-width:0!important;padding-bottom:8px!important}#chatHeader .info strong{display:block!important;font-size:17px!important;line-height:1.15!important;font-weight:900!important;letter-spacing:0!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important}#chatHeader .info span{display:flex!important;align-items:center!important;gap:6px!important;font-size:12px!important;opacity:.9!important;margin-top:4px!important}#chatHeader .info span:before{content:"";width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 4px rgba(34,197,94,.18)}#chatClose{margin-left:auto!important;background:rgba(255,255,255,.14)!important;border:1px solid rgba(255,255,255,.18)!important;color:#fff!important;font-size:22px!important;line-height:1!important;cursor:pointer!important;width:38px!important;height:38px!important;border-radius:14px!important;display:flex!important;align-items:center!important;justify-content:center!important}#chatClose:hover{background:rgba(255,255,255,.24)!important}#chatMessages{flex:1!important;overflow-y:auto!important;padding:18px!important;display:flex!important;flex-direction:column!important;gap:12px!important;max-height:none!important;background:linear-gradient(180deg,#f8fafc 0%,#eef6ff 100%)!important;scroll-behavior:smooth!important}#chatMessages::-webkit-scrollbar{width:8px}#chatMessages::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:20px}.msg{max-width:88%!important;padding:12px 14px!important;border-radius:18px!important;font-size:13px!important;line-height:1.58!important;letter-spacing:0!important;box-shadow:0 8px 22px rgba(15,23,42,.08)!important;word-break:break-word!important}.msg.bot{background:#fff!important;color:#111827!important;border:1px solid #e5e7eb!important;border-bottom-left-radius:7px!important;align-self:flex-start!important}.msg.user{background:linear-gradient(135deg,#2563eb,#1d4ed8)!important;color:#fff!important;border-bottom-right-radius:7px!important;align-self:flex-end!important}.msg.typing{opacity:.7!important}.msg b{font-weight:900}.quick-chips{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr))!important;gap:8px!important;padding:12px 14px!important;background:#fff!important;border-top:1px solid #e5e7eb!important}.chip{background:#f8fafc!important;border:1px solid #dbe4f0!important;color:#1f2937!important;padding:9px 10px!important;border-radius:14px!important;font-size:12px!important;font-weight:800!important;cursor:pointer!important;font-family:inherit!important;transition:background .15s,border-color .15s,transform .15s!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important}.chip:hover{background:#eff6ff!important;border-color:#2563eb!important;color:#1d4ed8!important;transform:translateY(-1px)!important}#chatInputWrap{padding:14px!important;border-top:1px solid #e5e7eb!important;display:flex!important;gap:10px!important;background:#fff!important}#chatInput{flex:1!important;min-width:0!important;padding:13px 14px!important;border:1.5px solid #cbd5e1!important;border-radius:16px!important;font-size:14px!important;background:#f8fafc!important;color:#111827!important;font-family:inherit!important;outline:none!important;transition:border-color .15s,box-shadow .15s,background .15s!important}#chatInput:focus{border-color:#2563eb!important;box-shadow:0 0 0 4px rgba(37,99,235,.12)!important;background:#fff!important}#chatSend{background:linear-gradient(135deg,#2563eb,#0f766e)!important;color:#fff!important;border:none!important;border-radius:16px!important;width:52px!important;min-width:52px!important;height:48px!important;padding:0!important;cursor:pointer!important;font-size:0!important;font-weight:900!important;box-shadow:0 10px 24px rgba(37,99,235,.24)!important}#chatSend:before{content:"➤";font-size:18px}#chatSend:hover{filter:brightness(1.06)!important}.vitals-chat-tools{display:flex;align-items:center;gap:7px;margin-top:8px;flex-wrap:wrap}.vitals-pill{display:inline-flex;align-items:center;gap:5px;border:1px solid #dbeafe;background:#eff6ff;color:#1d4ed8;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:800}.vitals-risk-meter{height:8px;border-radius:999px;background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444);margin:8px 0 2px;position:relative;overflow:hidden}.vitals-risk-meter span{position:absolute;top:0;bottom:0;width:4px;background:#111827;border-radius:999px}@keyframes vitalsChatIn{from{opacity:0;transform:translateY(10px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}@media(max-width:560px){#chatBubble{right:14px!important;bottom:14px!important}#chatBox{width:calc(100vw - 20px)!important;height:calc(100vh - 96px)!important;border-radius:20px!important}#chatHeader{padding:16px!important}.quick-chips{grid-template-columns:1fr 1fr!important}.chip{font-size:11px!important;padding:8px!important}#chatToggle{width:58px!important;height:58px!important;border-radius:20px!important}}@media(prefers-color-scheme:dark){#chatBox{background:rgba(15,23,42,.96)!important;border-color:#334155!important}#chatMessages{background:linear-gradient(180deg,#0f172a,#111827)!important}.msg.bot{background:#1e293b!important;color:#f8fafc!important;border-color:#334155!important}.quick-chips,#chatInputWrap{background:#0f172a!important;border-color:#334155!important}.chip{background:#1e293b!important;color:#e5e7eb!important;border-color:#334155!important}#chatInput{background:#111827!important;color:#f8fafc!important;border-color:#334155!important}}';
    document.head.appendChild(style);
  }

  function polishExistingChatUi(){
    const headerInfo = document.querySelector('#chatHeader .info span');
    if (headerInfo) headerInfo.textContent = 'AI prediction · Explanations · Health info';
    const input = document.getElementById('chatInput');
    if (input) input.placeholder = 'Ask for prediction, explanation, symptoms...';
    const chips = document.querySelectorAll('.quick-chips .chip');
    const labels = ['Predict my risk','Explain result','Warning symptoms','Lifestyle tips','General health'];
    chips.forEach((chip, i) => { if (labels[i]) chip.textContent = labels[i]; });
  }

  function ensureUi(){
    applyAdvancedChatUi();
    if (document.getElementById('chatBubble')) {
      polishExistingChatUi();
      return;
    }
    const root = document.createElement('div');
    root.id = 'chatBubble';
    root.innerHTML = '<div id="chatBox"><div id="chatHeader"><div class="avatar">AI</div><div class="info"><strong>VitalsAI Assistant</strong><span>AI prediction · Explanations · Health info</span></div><button id="chatClose" onclick="toggleChat()">×</button></div><div id="chatMessages"></div><div class="quick-chips"><button class="chip">Predict my risk</button><button class="chip">Explain result</button><button class="chip">Warning symptoms</button><button class="chip">Lifestyle tips</button><button class="chip">General health</button></div><div id="chatInputWrap"><input id="chatInput" placeholder="Ask for prediction, explanation, symptoms..." onkeydown="if(event.key===\'Enter\')sendChat()"><button id="chatSend" onclick="sendChat()">Send</button></div></div><button id="chatToggle" onclick="toggleChat()">AI</button>';
    document.body.appendChild(root);
  }

  if (typeof window.addMsg !== 'function') {
    window.addMsg = function(role, text){
      ensureUi();
      const el = document.getElementById('chatMessages');
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.innerHTML = text;
      el.appendChild(div);
      el.scrollTop = el.scrollHeight;
      return div;
    };
  }

  function ensureWelcome(){
    ensureUi();
    const messages = document.getElementById('chatMessages');
    if (!messages || messages.dataset.dynamicReady) return;
    messages.dataset.dynamicReady = '1';
    const disease = currentDisease();
    messages.innerHTML = '<div class="msg bot"><b>VitalsAI Dynamic Assistant</b><br>I can use form values to provide a rough prediction, risk explanation, result summary, and general health information.<br><br>Ask: <b>predict my risk</b>, <b>explain my result</b>, <b>what should I do for fever?</b>, or anything about <b>'+disease+'</b>.</div>';
  }

  window.getDynamicVitalsAnswer = answer;
  window.askQuick = function(q){
    const input = document.getElementById('chatInput');
    if (input) input.value = q;
    window.sendChat();
  };

  function recentChatHistory(){
    const msgs = [...document.querySelectorAll('#chatMessages .msg')].slice(-8);
    return msgs.map(m => ({
      role: m.classList.contains('user') ? 'user' : 'assistant',
      content: m.innerText.trim().slice(0, 500)
    })).filter(m => m.content);
  }

  function buildChatContext(){
    return {
      page: currentDisease(),
      pathname: location.pathname,
      title: document.title,
      form_values: formEntries().map(e => ({label:e.label, value:e.value, raw:e.raw})),
      visible_result: resultSnapshot(),
      local_prediction: predictFromForm()
    };
  }

  async function askBackendAI(question){
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        message: question,
        mode: 'advanced',
        context: buildChatContext(),
        history: recentChatHistory()
      })
    });
    if (!res.ok) throw new Error('Chat API failed');
    const data = await res.json();
    if (!data || !data.response) throw new Error('Empty chat response');
    return data;
  }

  window.sendChat = function(){
    ensureWelcome();
    const inp = document.getElementById('chatInput');
    if (!inp) return;
    const q = inp.value.trim();
    if (!q) return;
    inp.value = '';
    addMsg('user', q);
    const typing = addMsg('bot', '<span class="typing">Thinking dynamically...</span>');
    setTimeout(async () => {
      try {
        const backend = await askBackendAI(q);
        typing.innerHTML = backend.response;
      } catch (err) {
        typing.innerHTML = answer(q);
      }
      const box = document.getElementById('chatMessages');
      if (box) box.scrollTop = box.scrollHeight;
    }, 350);
  };

  const oldToggle = window.toggleChat;
  window.toggleChat = function(){
    if (typeof oldToggle === 'function') oldToggle();
    else document.getElementById('chatBox')?.classList.toggle('open');
    ensureWelcome();
  };

  document.addEventListener('DOMContentLoaded', () => {
    ensureUi();
    ensureWelcome();
    document.querySelectorAll('.quick-chips .chip').forEach((btn, i) => {
      const labels = ['Predict my risk','Explain result','Warning symptoms?','Lifestyle tips','General health info'];
      btn.onclick = () => window.askQuick(labels[i] || btn.textContent);
    });
  });
})();
