# Prompts

### Prompt Templates

- **Basic-1**
  - Correct the text:\n\n `{{TEXT}}`

- **Basic-2**
  - Correct the spelling and grammar of the following text:\n\n `{{TEXT}}`

- **Complex-1**
  - Correct the spelling and grammar of the following incorrect text from an optical character recognition (OCR) applied to a historical document:\n Incorrect text: `{{TEXT}}`\n The corrected text is:

- **Complex-2**
  - Please assist with reviewing and correcting errors in texts produced by automatic transcription (OCR) of historical documents. Your task is to carefully examine the following text and correct any mistakes introduced by the OCR software. The text to correct appears after the segment "TEXT TO CORRECT:". Please place the corrected version of the text after the "CORRECTED TEXT:" segment. Do not write anything else than the corrected text.\n\n TEXT TO CORRECT: `{{TEXT}}` \n CORRECTED TEXT:

- **Complex-3**
  - Complex-2 translated to fr, de, etc.

---
