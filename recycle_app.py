import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

st.set_page_config(
    page_title="ReCycle App",
    page_icon="♻️",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': '''
        <meta name="google-site-verification" content="7QcAzq0PmjFkL7r0lwBMQCVZgB6NMKBasyZsE_G7xKI" />
        '''
    }
)

MODEL_PATH = "plastic_classifier_model.keras"

plastic_full_names = {
    "PET": "PET (Polyethylene Terephthalate)",
    "HDPE": "HDPE (High-Density Polyethylene)",
    "PVC": "PVC (Polyvinyl Chloride)",
    "LDPE": "LDPE (Low-Density Polyethylene)",
    "PP": "PP (Polypropylene)",
    "PS": "PS (Polystyrene)"
}

class_names = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

recycling_ideas = {
    "English": {
        "PET": [
            ("You can upcycle this bottle into a self-watering planter by cutting it in half and adding a cotton wick.", "https://img.icons8.com/color/96/potted-plant.png"),
            ("It can be repurposed as a neat pencil holder for your study table or desk.", "https://img.icons8.com/color/96/pencil-holder.png"),
            ("You can turn it into a simple piggy bank by making a small slot on top for saving coins.", "https://img.icons8.com/color/96/piggy-bank.png")
        ],
        "HDPE": [
            ("This sturdy container can be used to store household detergents, liquids, or cleaning supplies.", "https://img.icons8.com/color/96/cleaning-products.png"),
            ("You can transform it into a durable watering can for your home garden by drilling small holes in the cap.", "https://img.icons8.com/color/96/watering-can.png"),
            ("It can be cut and crafted into practical desk organizers for stationary items.", "https://img.icons8.com/color/96/desk.png")
        ],
        "PVC": [
            ("These pipes can be converted into custom desk organizers for tools and office accessories.", "https://img.icons8.com/color/96/toolbox.png"),
            ("You can use them as flexible cable managers to neatly organize cluttered wires behind your desk.", "https://img.icons8.com/color/96/cable-release.png"),
            ("They can be used as durable border edging for garden beds and outdoor pathways.", "https://img.icons8.com/color/96/fence.png")
        ],
        "LDPE": [
            ("You can weave clean plastic bags together into 'plarn' (plastic yarn) to make reusable shopping bags.", "https://img.icons8.com/color/96/shopping-bag.png"),
            ("They can be fused together using an iron to create waterproof mats or protective covers.", "https://img.icons8.com/color/96/mat.png")
        ],
        "PP": [
            ("You can chop colorful caps into small tiles to create creative mosaic art projects.", "https://img.icons8.com/color/96/art.png"),
            ("These containers can be cut and fitted as custom drawer dividers for small household items.", "https://img.icons8.com/color/96/drawer.png"),
            ("They make excellent small seed starter pots for growing new plants at home.", "https://img.icons8.com/color/96/sprout.png")
        ],
        "PS": [
            ("You can craft foam or rigid polystyrene pieces into lightweight picture frames.", "https://img.icons8.com/color/96/picture-frame.png"),
            ("It can be reused to build durable architectural or hobby models for DIY projects.", "https://img.icons8.com/color/96/craft.png")
        ]
    },
    "Tamil": {
        "PET": [
            ("இந்த பாட்டிலை பாதியாக வெட்டி, பஞ்சு திரி சேர்த்து தானாக நீர் பாய்ச்சும் செடி தொட்டியாக மாற்றலாம்.", "https://img.icons8.com/color/96/potted-plant.png"),
            ("உங்கள் படிப்பு மேஜைக்கு அழகான பென்சில் ஹோல்டராக பயன்படுத்தலாம்.", "https://img.icons8.com/color/96/pencil-holder.png"),
            ("மேல் பகுதியில் சிறிய துளையிட்டு நாணயங்களை சேமிக்கும் உண்டியலாக மாற்றலாம்.", "https://img.icons8.com/color/96/piggy-bank.png")
        ],
        "HDPE": [
            ("இந்த உறுதியான பாட்டிலை துப்புரவு திரவங்கள் அல்லது சோப்பு திரவங்களை சேமிக்க பயன்படுத்தலாம்.", "https://img.icons8.com/color/96/cleaning-products.png"),
            ("மூடியில் சிறிய துளைகள் போட்டு தோட்டத்திற்கு நீர் பாய்ச்சும் பூவாளியாக மாற்றலாம்.", "https://img.icons8.com/color/96/watering-can.png"),
            ("பொருட்களை அடுக்க வைக்கும் மேஜை அமைப்பாக வெட்டி பயன்படுத்தலாம்.", "https://img.icons8.com/color/96/desk.png")
        ],
        "PVC": [
            ("இந்த பைப்புகளை அலுவலக பொருட்கள் மற்றும் கருவிகளை வைக்கும் அமைப்பாக மாற்றலாம்.", "https://img.icons8.com/color/96/toolbox.png"),
            ("மேஜை பின்னால் உள்ள ஒயர்களை ஒழுங்கமைக்க கேபிள் மேனேஜராக பயன்படுத்தலாம்.", "https://img.icons8.com/color/96/cable-release.png"),
            ("தோட்டத்து பாதைகளுக்கு உறுதியான தடுப்பு வேலியாக பயன்படுத்தலாம்.", "https://img.icons8.com/color/96/fence.png")
        ],
        "LDPE": [
            ("பிளாஸ்டிக் பைகளை நெய்து மீண்டும் பயன்படுத்தக்கூடிய ஷாப்பிங் பைகளாக மாற்றலாம்.", "https://img.icons8.com/color/96/shopping-bag.png"),
            ("அயர்ன் பாக்ஸ் மூலம் சூடுபடுத்தி நீர்ப்புகா பாய்களாக தயாரிக்கலாம்.", "https://img.icons8.com/color/96/mat.png")
        ],
        "PP": [
            ("வண்ண மூடிகளை சிறு துண்டுகளாக வெட்டி கலைப் பொருட்களாக மாற்றலாம்.", "https://img.icons8.com/color/96/art.png"),
            ("டிராயர்களில் சிறிய பொருட்களை தனித்தனியாக பிரிக்க பயன்படுத்தலாம்.", "https://img.icons8.com/color/96/drawer.png"),
            ("வீட்டில் புதிய செடிகளை வளர்க்க விதை தொட்டியாக பயன்படுத்தலாம்.", "https://img.icons8.com/color/96/sprout.png")
        ],
        "PS": [
            ("தெர்மோகோல் துண்டுகளை லேசான போட்டோ பிரேம்களாக மாற்றலாம்.", "https://img.icons8.com/color/96/picture-frame.png"),
            ("கட்டிட மாதிரிகள் அல்லது கைவினை திட்டங்களுக்கு பயன்படுத்தலாம்.", "https://img.icons8.com/color/96/craft.png")
        ]
    },
    "Malayalam": {
        "PET": [
            ("ഈ കുപ്പി പകുതിയായി മുറിച്ച് പഞ്ഞി തിരിയിട്ട് സ്വയം വെള്ളമൊഴിക്കുന്ന ചെടിത്തൊട്ടിയാക്കാം.", "https://img.icons8.com/color/96/potted-plant.png"),
            ("പഠനമേശയിൽ പെൻസിലുകൾ വെക്കാനുള്ള പാത്രമായി ഉപയോഗിക്കാം.", "https://img.icons8.com/color/96/pencil-holder.png"),
            ("മുകളിൽ ചെറിയ ദ്വാരമിട്ട് നാണയങ്ങൾ സൂക്ഷിക്കുന്ന സിപ്പിയാക്കാം.", "https://img.icons8.com/color/96/piggy-bank.png")
        ],
        "HDPE": [
            ("സോപ്പ് ലായനികളും ക്ലീനിംഗ് സാമഗ്രികളും സൂക്ഷിക്കാൻ ഇത് ഉപയോഗിക്കാം.", "https://img.icons8.com/color/96/cleaning-products.png"),
            ("അടപ്പിൽ ചെറിയ ദ്വാരങ്ങളിട്ട് ചെടികൾക്ക് വെള്ളമൊഴിക്കുന്ന പാത്രമാക്കാം.", "https://img.icons8.com/color/96/watering-can.png"),
            ("മേശപ്പുറത്തെ സാധനങ്ങൾ ഒതുക്കി വെക്കാനുള്ള ബോക്സുകളാക്കാം.", "https://img.icons8.com/color/96/desk.png")
        ],
        "PVC": [
            ("ടൂളുകളും ഓഫീസിലെ സാധനങ്ങളും വെക്കാനുള്ള ഹോൾഡറുകളാക്കാം.", "https://img.icons8.com/color/96/toolbox.png"),
            ("കേബിളുകൾ ഭംഗിയായി ഒതുക്കി വെക്കാൻ ഉപയോഗിക്കാം.", "https://img.icons8.com/color/96/cable-release.png"),
            ("തോട്ട അതിരുകൾ നിർമ്മിക്കാൻ ഉപയോഗിക്കാം.", "https://img.icons8.com/color/96/fence.png")
        ],
        "LDPE": [
            ("പ്ലാസ്റ്റിക് കവറുകൾ നെയ്ത് വീണ്ടും ഉപയോഗിക്കാവുന്ന ബാഗുകളാക്കാം.", "https://img.icons8.com/color/96/shopping-bag.png"),
            ("ഐൺ ബോക്സ് ഉപയോഗിച്ച് ചൂടാക്കി വാട്ടർപ്രൂഫ് മാറ്റുകളാക്കാം.", "https://img.icons8.com/color/96/mat.png")
        ],
        "PP": [
            ("കളർ അടപ്പുകൾ ചെറിയ കഷ്ണങ്ങളാക്കി ആർട്ട് പ്രൊജക്റ്റുകൾ ചെയ്യാം.", "https://img.icons8.com/color/96/art.png"),
            ("ഡ്രോയറുകളിൽ സാധനങ്ങൾ തരംതിരിച്ചു വെക്കാൻ ഉപയോഗിക്കാം.", "https://img.icons8.com/color/96/drawer.png"),
            ("ചെടികൾ മുളപ്പിച്ചെടുക്കാൻ ചെറിയ തൊട്ടികളായി ഉപയോഗിക്കാം.", "https://img.icons8.com/color/96/sprout.png")
        ],
        "PS": [
            ("തെർമോക്കോൾ കഷ്ണങ്ങൾ ഫോട്ടോ ഫ്രെയിമുകളാക്കി മാറ്റാം.", "https://img.icons8.com/color/96/picture-frame.png"),
            ("മോഡലുകളും ക്രാഫ്റ്റ് വസ്തുക്കളും ഉണ്ടാക്കാൻ ഉപയോഗിക്കാം.", "https://img.icons8.com/color/96/craft.png")
        ]
    },
    "Hindi": {
        "PET": [
            ("इस बोतल को आधा काटकर और कॉटन की बत्ती लगाकर सेल्फ-वॉटरिंग पॉट बनाएं।", "https://img.icons8.com/color/96/potted-plant.png"),
            ("इसे अपनी स्टडी टेबल के लिए पेंसिल होल्डर के रूप में इस्तेमाल करें।", "https://img.icons8.com/color/96/pencil-holder.png"),
            ("ऊपर एक छोटा छेद करके सिक्के बचाने के लिए गुल्लक बनाएं।", "https://img.icons8.com/color/96/piggy-bank.png")
        ],
        "HDPE": [
            ("इस मजबूत डिब्बे का उपयोग डिटर्जेंट या सफाई के सामान को रखने के लिए करें।", "https://img.icons8.com/color/96/cleaning-products.png"),
            ("ढक्कन में छोटे छेद करके अपने बगीचे के लिए वॉटरिंग कैन बनाएं।", "https://img.icons8.com/color/96/watering-can.png"),
            ("इसे काटकर टेबल ऑर्गनाइज़र के रूप में उपयोग करें।", "https://img.icons8.com/color/96/desk.png")
        ],
        "PVC": [
            ("इन पाइपों को औजारों और ऑफिस सामान के ऑर्गनाइज़र में बदलें।", "https://img.icons8.com/color/96/toolbox.png"),
            ("टेबल के पीछे बिखरे तारों को व्यवस्थित करने के लिए उपयोग करें।", "https://img.icons8.com/color/96/cable-release.png"),
            ("बगीचे की क्यारियों के लिए मजबूत बॉर्डर बनाएं।", "https://img.icons8.com/color/96/fence.png")
        ],
        "LDPE": [
            ("प्लास्टिक बैग्स को बुनकर दोबारा इस्तेमाल योग्य शॉपिंग बैग बनाएं।", "https://img.icons8.com/color/96/shopping-bag.png"),
            ("प्रेस (Iron) से जोड़कर वॉटरप्रूफ मैट बनाएं।", "https://img.icons8.com/color/96/mat.png")
        ],
        "PP": [
            ("रंगीन ढक्कनों को छोटे टुकड़ों में काटकर आर्ट प्रोजेक्ट बनाएं।", "https://img.icons8.com/color/96/art.png"),
            ("दराजों (Drawers) में छोटे सामान को अलग रखने के लिए उपयोग करें।", "https://img.icons8.com/color/96/drawer.png"),
            ("पौधे उगाने के लिए छोटे गमलों के रूप में उपयोग करें।", "https://img.icons8.com/color/96/sprout.png")
        ],
        "PS": [
            ("थर्माकोल के टुकड़ों से हल्के फोटो फ्रेम बनाएं।", "https://img.icons8.com/color/96/picture-frame.png"),
            ("डीआईवाई प्रोजेक्ट्स और क्राफ्ट मॉडल बनाने में उपयोग करें।", "https://img.icons8.com/color/96/craft.png")
        ]
    },
    "Telugu": {
        "PET": [
            ("ఈ సీసాని సగానికి కోసి, కాటన్ వత్తిని అమర్చి సెల్ఫ్-వాటరింగ్ ప్లాంటర్‌గా మార్చవచ్చు.", "https://img.icons8.com/color/96/potted-plant.png"),
            ("మీ స్టడీ టేబుల్ కోసం పెన్సిల్ హోల్డర్‌గా ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/pencil-holder.png"),
            ("నాణేలు దాచుకోవడానికి చిన్న రంధ్రం చేసి పిగ్గీ బ్యాంక్‌గా మార్చవచ్చు.", "https://img.icons8.com/color/96/piggy-bank.png")
        ],
        "HDPE": [
            ("ఈ గట్టి బాటిల్‌ను డిటర్జెంట్లు లేదా లిక్విడ్‌లను దాచడానికి ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/cleaning-products.png"),
            ("మూతకి చిన్న రంధ్రాలు చేసి మొక్కలకు నీళ్లు పోసే కాన్‌గా మార్చవచ్చు.", "https://img.icons8.com/color/96/watering-can.png"),
            ("టేబుల్ ఆర్గనైజర్‌గా కట్ చేసి ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/desk.png")
        ],
        "PVC": [
            ("ఈ పైపులను ఆఫీస్ టూల్స్ మరియు వైర్లు అమర్చుకోవడానికి ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/toolbox.png"),
            ("కేబుల్స్ సరిగ్గా అమర్చడానికి కేబుల్ మేనేజర్‌గా ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/cable-release.png"),
            ("గార్డెన్ సరిహద్దుల కోసం ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/fence.png")
        ],
        "LDPE": [
            ("ప్లాస్టిక్ కవర్లను అల్లి షాపింగ్ బ్యాగులుగా తయారు చేయవచ్చు.", "https://img.icons8.com/color/96/shopping-bag.png"),
            ("ఐరన్ బాక్స్‌తో వేడి చేసి వాటర్‌ప్రూఫ్ మ్యాట్‌లుగా మార్చవచ్చు.", "https://img.icons8.com/color/96/mat.png")
        ],
        "PP": [
            ("రంగు మూతలను చిన్న ముక్కలుగా చేసి ఆర్ట్ ప్రాజెక్ట్‌లు చేయవచ్చు.", "https://img.icons8.com/color/96/art.png"),
            ("డ్రాయర్లలో చిన్న వస్తువులను సర్దుకోవడానికి ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/drawer.png"),
            ("చిన్న మొక్కలు పెంచడానికి కుండీలుగా ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/sprout.png")
        ],
        "PS": [
            ("థర్మోకోల్ ముక్కలతో తేలికపాటి ఫోటో ఫ్రేమ్‌లు చేయవచ్చు.", "https://img.icons8.com/color/96/picture-frame.png"),
            ("క్రాఫ్ట్ మోడల్స్ తయారు చేయడానికి ఉపయోగించవచ్చు.", "https://img.icons8.com/color/96/craft.png")
        ]
    },
    "Kannada": {
        "PET": [
            ("ಈ సీసೆಯನ್ನು ಅರ್ಧಕ್ಕೆ ಕತ್ತರಿಸಿ, ಹತ್ತಿ ಬತ್ತಿ ಹಾಕಿ ಸ್ವಯಂ-ನೀರೆರೆಯುವ ಕುಂಡವಾಗಿ ಮಾಡಬಹುದು.", "https://img.icons8.com/color/96/potted-plant.png"),
            ("ನಿಮ್ಮ ಅಧ್ಯಯನ ಮೇಜಿಗೆ ಪೆನ್ಸಿಲ್ ಹೋಲ್ಡರ್ ಆಗಿ ಬಳಸಿ.", "https://img.icons8.com/color/96/pencil-holder.png"),
            ("ಮೇಲ್ಭಾಗದಲ್ಲಿ ಸಣ್ಣ ರಂಧ್ರ ಮಾಡಿ ನಾಣ್ಯ ಉಳಿಸುವ ގುಲ್ಲಕ್ ಮಾಡಬಹುದು.", "https://img.icons8.com/color/96/piggy-bank.png")
        ],
        "HDPE": [
            ("ಈ ಗಟ್ಟಿ ಬಾಟಲಿಯನ್ನು ಡಿಟರ್ಜೆಂಟ್ ಅಥವಾ ಕ್ಲೀನಿಂಗ್ ದ್ರವಗಳನ್ನು ಇಡಲು ಬಳಸಿ.", "https://img.icons8.com/color/96/cleaning-products.png"),
            ("ಮೂಡಿಗೆ ಸಣ್ಣ ರಂಧ್ರ ಮಾಡಿ ಗಿಡಗಳಿಗೆ ನೀರು ಹಾಕುವ ಕ್ಯಾನ್ ಮಾಡಬಹುದು.", "https://img.icons8.com/color/96/watering-can.png"),
            ("ಡೆಸ್ಕ್ ആర్గನೈಸರ್ ಆಗಿ ಕತ್ತರಿಸಿ ಬಳಸಬಹುದು.", "https://img.icons8.com/color/96/desk.png")
        ],
        "PVC": [
            ("ಈ ಪೈಪ್‌ಗಳನ್ನು ಆಫೀಸ್ ಉಪಕರಣಗಳನ್ನು ಇಡುವ ಹೋಲ್ಡರ್ ಆಗಿ ಮಾರ್ಪಡಿಸಿ.", "https://img.icons8.com/color/96/toolbox.png"),
            ("ವೈರ್‌ಗಳನ್ನು ನೀಟಾಗಿ ಜೋಡಿಸಲು ಕೇಬಲ್ మేనేజర్ ಆಗಿ ಬಳಸಿ.", "https://img.icons8.com/color/96/cable-release.png"),
            ("ತೋಟದ ಗಡಿಗಳಿಗೆ ತಡೆಯಾಗಿ ಬಳಸಬಹುದು.", "https://img.icons8.com/color/96/fence.png")
        ],
        "LDPE": [
            ("ಪ್ಲಾಸ್ಟಿಕ್ ಚೀಲಗಳನ್ನು ಹೆಣೆದು ಮರುಬಳಕೆಯ શોಪಿಂಗ್ ಬ್ಯಾಗ್ ಮಾಡಬಹುದು.", "https://img.icons8.com/color/96/shopping-bag.png"),
            ("ఐరన్ బాక్స్ ಬಳಸಿ ಬಿಸಿ ಮಾಡಿ ವಾಟರ್‌ಪ್ರೂಫ್ ಮ್ಯಾಟ್‌ಗಳನ್ನು ಮಾಡಬಹುದು.", "https://img.icons8.com/color/96/mat.png")
        ],
        "PP": [
            ("ಬಣ್ಣದ ಮುಚ್ಚಳಗಳನ್ನು ಸಣ್ಣ ತುಂಡುಗಳಾಗಿ ಕತ್ತರಿಸಿ ಆರ್ಟ್ ಪ್ರಾಜೆಕ್ಟ್ ಮಾಡಬಹುದು.", "https://img.icons8.com/color/96/art.png"),
            ("ಡ್ರಾಯರ್‌ಗಳಲ್ಲಿ ವಸ್ತುಗಳನ್ನು ಪ್ರತ್ಯೇಕಿಸಲು ಬಳಸಿ.", "https://img.icons8.com/color/96/drawer.png"),
            ("ಸಣ್ಣ ಗಿಡಗಳನ್ನು ಬೆಳೆಸಲು ಕುಂಡಗಳಾಗಿ ಬಳಸಿ.", "https://img.icons8.com/color/96/sprout.png")
        ],
        "PS": [
            ("ಥರ್ಮೋಕೋಲ್ ತುಂಡುಗಳಿಂದ ಹಗುರವಾದ ಫೋಟೋ ಫ್ರೇಮ್‌ಗಳನ್ನು ಮಾಡಬಹುದು.", "https://img.icons8.com/color/96/picture-frame.png"),
            ("ಕ್ರಾಫ್ಟ್ ಮಾಡೆಲ್‌ಗಳನ್ನು ಮಾಡಲು ಬಳಸಿ.", "https://img.icons8.com/color/96/craft.png")
        ]
    }
}

@st.cache_resource
def load_keras_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None

model = load_keras_model()

st.title("♻️ ReCycle")

selected_language = st.selectbox(
    "🌐 Choose Language / மொழியைத் தேர்ந்தெடுக்கவும்:",
    ["English", "Tamil", "Malayalam", "Hindi", "Telugu", "Kannada"]
)

st.write("Upload or capture an image of a plastic item to predict its type and get recycling ideas.")

uploaded_file = st.file_uploader(
    "Choose a plastic image...", 
    type=["jpg", "jpeg", "png", "bmp"]
)

camera_file = st.camera_input("Or take a photo using your camera")

input_image_file = camera_file if camera_file is not None else uploaded_file

if input_image_file is not None and model is not None:
    image = Image.open(input_image_file).convert('RGB')
    st.image(image, caption="Selected Image", use_container_width=True)
    
    with st.spinner("Classifying plastic type..."):
        try:
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            predicted_short = class_names[predicted_idx]
            predicted_full = plastic_full_names.get(predicted_short, predicted_short)
            confidence = float(np.max(predictions[0])) * 100

            st.success(f"**Predicted Plastic Type:** {predicted_full}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            st.subheader(f"💡 Recycling Ideas for {predicted_full}:")
            
            lang_ideas = recycling_ideas.get(selected_language, recycling_ideas["English"])
            suggestions = lang_ideas.get(predicted_short, [("No suggestions found.", "")])
            
            for text, img_url in suggestions:
                col1, col2 = st.columns([1, 5])
                with col1:
                    if img_url:
                        st.image(img_url, width=50)
                with col2:
                    st.write(f"- {text}")

        except Exception as e:
            st.error(f"Error processing image: {e}")
