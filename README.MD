# Links
* [Dev Day Announcement](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)
* [Pricing](https://openai.com/pricing)
* [Assistants API](https://platform.openai.com/docs/assistants/overview)
* [Vision API](https://platform.openai.com/docs/guides/vision)

## Pricing comparison

| type       | gpt3.5-turbo | gpt4-turbo | gpt4  | ada-002 |
|------------|--------------|------------|-------|---------|
| prompt     | 0.0010       | 0.01       | 0.03+ | 0.0001  |
| completion | 0.0020       | 0.03       | 0.06+ | -       |

- ada-002 is the embedding model


# API Key

Please create the `.env` file in the project folder with content:
```
OPENAI_API_KEY=""
```
[get OpenAI key](https://platform.openai.com/account/api-keys) press "➕ Create new..." and paste the value


# Installation
```shell
pip install -r requirements.txt
```
[setting up venv](https://gist.github.com/ton77v/cd625ca0b8ee7b4bd3a67791a4fce903) is recommended

# Usage

## Assistants API
> Any questions to the TankOps / MarineOps Tutor

- could use [Playground](https://platform.openai.com/playground?mode=assistant) 
to do the test runs from the UI
- although need to create the Assistant & upload the sample file first!

When no Assistant:
```shell
python assistants_api.py -q "What are the frequently used storage tank standards and practices referenced in 25 PA Code, Chapter 245 by National Leak Prevention Association?"
```

With existing Assistant:
```shell
python assistants_api.py -q "What are the frequently used storage tank standards and practices referenced in 25 PA Code, Chapter 245 by National Leak Prevention Association?" -a "asst_fFp2EhDiwKh7bfsAdnO2karf"
```

Follow up Question to existing Thread:
```shell
python assistants_api.py -q "What was the previous question?" -a "asst_fFp2EhDiwKh7bfsAdnO2karf" -t "thread_uauuFqLi8ppplEUjvldRagQW"
```

Finally, let's see if the Functions Calling works:
```shell
python assistants_api.py -q "What's the tech specification for storage tank XX-88-1080-QQ?" -a "asst_fFp2EhDiwKh7bfsAdnO2karf"
```


## Vision API

Question about Hot Water Storage Tank | anything related to the tank storage facilities, port marine terminals operations, etc
```shell
python vision_api.py -q "What is it and what are the specifications?" -i "C:\Users\expor\PYTHON\TRYING_STUFF\open_ai_dev_day_2023\images\HotWaterTank.png"
```

Questions about anything else
```shell
python vision_api.py -q "What is it?" -i "C:\Users\expor\PYTHON\TRYING_STUFF\open_ai_dev_day_2023\images\RAG.png" --use-generic-prompt
```

# Creating Custom GPT

1. ➕ create GPT in App | [via browser like so](https://chat.openai.com/gpts/editor) requires Plus subscription

2. Paste:
```
Create the world-class tutor for the students learning such matters as tank storage facility management and operations, port marine terminals operations for oil and general liquid bulk operations, etc. The mission is to help students and operators better understand their operations.
```

3. Talk with GPT Builder about the name and profile picture

4. Refine the topics | areas of expertise; alternatively:
```
Do you have any suggestions?
```
And accept the suggestions

5. Builder will ask "would u like to try..." but first let's check the "Configure" tab above:

## Configure

We could **modify the Prompt** there
📝 that using the Builder could be useful to get ideas while engineering the prompts, I got this:
```
As the 'Storage Marine Operations Tutor,' your role is to provide expert guidance in tank storage facility management and marine terminal operations. Focus on topics like the fundamentals of tank storage, detailed safety practices, operational procedures, environmental regulations, emergency response strategies, and the latest technologies and trends in the field. Your responses should be educational, accurate, and tailored to learners at various levels, from beginners to experienced professionals. Emphasize practical application and safety, offering step-by-step explanations and staying up-to-date with industry standards and advancements.
```

Note the **Conversation starters** that's what users see right when connecting to the GPT

Could **Upload files** to the Knowledge base.
> ...you can upload a maximum of **three files** that the GPT can reference in its responses. These files could include documents, spreadsheets, images, or other relevant materials

By default the GPT will have
- Web Browsing Access
- DALL-E Image Generation
- if need Code Interpreter (Python), should check this as well.

Actions:
- Add actions button allows importing OpenAPI JSON schema describing the API endpoints & the actual endpoints GPT could use;
it's pretty similar to defining the tool with the Assistants API
- This way we could get access to our Databases! Although we wouldn't know which user requests the data

6. Now can try a few questions in the Preview on the right side (Builder calls it a playground)
- Share feedback with the Builder if notice something you'd like to change.

7. Save button above and choose between:
- Only me
- Only people with a link
- Public | eventually this *may* list the GPT in the Store once it will be started

Could add the Website URL to the Builder Profile using 🖋 "edit"

8. After that it will appear above the conversations in the sidebar; to grab the link use the dropdown menu 
> the menu expands from the GPT name aside the sidebar

Link will look like:
```
https://chat.openai.com/g/g-jHqqscnkw-storage-marine-operations-tutor
```
- to use this, the user should have a Plus subscription to the ChatGPT
- it won't work outside ChatGPT website and the "Custom GPT" can't be accessed programmatically

⚙ Edit GPT option is available from the dropdown menu: we'll access the GPT Builder, Configure and the Preview

# Images

ES-48-1080-H 1080 Gallon Carbon Steel
Horizontal Epoxy Lined Hot Water Tank
160 psig, ASME Section IV, HLW
HORIZONTAL HOT WATER STORAGE TANKS
Hanson Pressure Tanks Finished Projects

[source](https://hansontank.com/horizontal-epoxy-lined-hot-water-tanks/es481080h/)

Tank Specs
Volume: 1080 gallons
Size: 48″ OD x 148″ OA
Design Pressure: 160 psig
Orientation: Horizontal
Contents: Hot Water
Service: Storage
Material: Carbon Steel
Empty Weight: 1600 lbs
Lining: Epoxy
Exterior: Prime
Insulation: None
Code: ASME Section IV, HLW

# Docs

> listing_of_standards_and_practices.pdf

List of Frequently utilized Tank Standards and Practices
[source](https://files.dep.state.pa.us/environmentalcleanupbrownfields/storagetanks/storagetanksportalfiles/abovegroundst/listing_of_standards_and_practices.pdf)

What are the frequently used storage tank standards and practices referenced in 25 PA Code, 
Chapter 245 by National Leak Prevention Association?


# contact the developer  
```  
antoniov77v@gmail.com  
```  
[other contacts](https://about.me/antonio-v/)  

