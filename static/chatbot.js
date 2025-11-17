// static/chatbot.js
// Handles sending user message to Flask backend and rendering messages.

const chatBox = () => document.querySelector(".messages");
const inputBox = () => document.querySelector("#chatInput");
const sendBtn = () => document.querySelector("#sendBtn");

// append message helper
function appendMessage(text, who) {
  const d = document.createElement("div");
  d.className = `msg ${who}`;
  d.innerText = text;
  chatBox().appendChild(d);
  chatBox().scrollTop = chatBox().scrollHeight;
}

// show a temporary "typing" bubble
function showTyping() {
  const t = document.createElement("div");
  t.className = "msg bot";
  t.id = "typingBubble";
  t.innerText = "typing...";
  chatBox().appendChild(t);
  chatBox().scrollTop = chatBox().scrollHeight;
}

function removeTyping() {
  const el = document.getElementById("typingBubble");
  if (el) el.remove();
}

// send message to server
async function sendMessage() {
  const text = inputBox().value.trim();
  if (!text) return;
  appendMessage(text, "user");
  inputBox().value = "";
  showTyping();

  try {
    const res = await fetch("/chat_api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });

    if (!res.ok) {
      removeTyping();
      appendMessage("Error: unable to contact server.", "bot");
      return;
    }

    const data = await res.json();
    removeTyping();

    // server may return array of bot messages
    if (data.replies && Array.isArray(data.replies)) {
      data.replies.forEach(r => appendMessage(r, "bot"));
    } else if (data.reply) {
      appendMessage(data.reply, "bot");
    }

    // if prediction included, show as extra message
    if (data.prediction) {
      appendMessage(`Prediction: ${data.prediction}`, "bot");
    }
  } catch (err) {
    removeTyping();
    appendMessage("Error: network or server issue.", "bot");
    console.error(err);
  }
}

if (sendBtn()) {
  sendBtn().addEventListener("click", sendMessage);
  document.addEventListener("keydown", function(e){
    if(e.key === "Enter" && document.activeElement === inputBox()) {
      e.preventDefault();
      sendMessage();
    }
  });
}
