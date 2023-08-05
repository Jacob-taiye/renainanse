
const chatbotConversation = document.getElementById('chatbot-conversation')



document.addEventListener('submit', (e) => {
    e.preventDefault()
    const userInput = document.getElementById('user-input')   
    // fetchReply()
    const newSpeechBubble = document.createElement('div')
    newSpeechBubble.classList.add('speech', 'speech-human')
    chatbotConversation.appendChild(newSpeechBubble)
    newSpeechBubble.textContent = userInput.value
    userInput.value = ''
    chatbotConversation.scrollTop = chatbotConversation.scrollHeight

    
}) 


function renderTypewriterText(text) {
    const newSpeechBubble = document.createElement('div')
    newSpeechBubble.classList.add('speech', 'speech-ai', 'blinking-cursor')
    chatbotConversation.appendChild(newSpeechBubble)
    let i = 0
    const interval = setInterval(() => {
        newSpeechBubble.textContent += text.slice(i-1, i)
        if (text.length === i) {
            clearInterval(interval)
            newSpeechBubble.classList.remove('blinking-cursor')
        }
        i++
        chatbotConversation.scrollTop = chatbotConversation.scrollHeight
    }, 50)
}

//  var document = document.getElementById('form')
//  var userInput = document.getElementById('user-input');
//  var newP = document.getElementById('newP')
//  var button = document.getElementById('submit-btn')


//  button.addEventListener('click', (e)=>{
//     e.preventDefault();

//     if(userInput.value){
//         newP.textContent = `Bola Hammed Tinubu!!"`
//         userInput.value="";

//     }
//  })