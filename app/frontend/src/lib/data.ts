export interface Language {
  code: string;
  name: string;
  nativeName: string;
  flag: string;
  region: string;
}

export interface Example {
  en: string;
  label: string;
}

export const LANGUAGES: Language[] = [
  { code: "hau", name: "Hausa",  nativeName: "Hausa",   flag: "🇳🇬", region: "Nigeria / Niger" },
  { code: "ibo", name: "Igbo",   nativeName: "Igbo",    flag: "🇳🇬", region: "Nigeria" },
  { code: "yor", name: "Yoruba", nativeName: "Yorùbá",  flag: "🇳🇬", region: "Nigeria / Benin" },
  { code: "wol", name: "Wolof",  nativeName: "Wolof",   flag: "🇸🇳", region: "Senegal / Gambia" },
  { code: "ewe", name: "Ewe",    nativeName: "Eʋegbe",  flag: "🇬🇭", region: "Ghana / Togo" },
  { code: "fon", name: "Fon",    nativeName: "Fon gbe", flag: "🇧🇯", region: "Benin" },
  { code: "twi", name: "Twi",    nativeName: "Twi",     flag: "🇬🇭", region: "Ghana" },
];

export const EXAMPLES: Record<string, Example[]> = {
  hau: [
    { en: "The market opens early every morning.", label: "Daily life" },
    { en: "Education is the foundation of national development.", label: "Education" },
    { en: "The farmer harvested a large crop this season.", label: "Agriculture" },
    { en: "Clean water is essential for good health.", label: "Health" },
  ],
  ibo: [
    { en: "The children played football after school.", label: "Daily life" },
    { en: "Our community came together to help each other.", label: "Community" },
    { en: "Water is essential for all living things.", label: "Nature" },
    { en: "The teacher praised the hardworking students.", label: "Education" },
  ],
  yor: [
    { en: "Lagos is a vibrant and busy city.", label: "Urban life" },
    { en: "We celebrate our culture with music and dance.", label: "Culture" },
    { en: "The rain has come, and the farmers are happy.", label: "Agriculture" },
    { en: "Knowledge passed down through generations is priceless.", label: "Tradition" },
  ],
  wol: [
    { en: "The sun rises in the east and sets in the west.", label: "Nature" },
    { en: "Farmers work hard to feed their families.", label: "Agriculture" },
    { en: "Peace is the foundation of development.", label: "Society" },
    { en: "A good neighbor is worth more than a distant relative.", label: "Proverb" },
  ],
  ewe: [
    { en: "The elders gathered to resolve the dispute.", label: "Community" },
    { en: "A good harvest brings joy to the whole village.", label: "Agriculture" },
    { en: "Children are the future of the nation.", label: "Education" },
    { en: "The river flows gently through the valley.", label: "Nature" },
  ],
  fon: [
    { en: "The river gives life to the surrounding land.", label: "Nature" },
    { en: "We must protect our forests and wildlife.", label: "Environment" },
    { en: "Knowledge is more valuable than gold.", label: "Proverb" },
    { en: "He cooked food for his entire family.", label: "Daily life" },
  ],
  twi: [
    { en: "The chief called a meeting of the elders.", label: "Governance" },
    { en: "Honesty and hard work lead to success.", label: "Values" },
    { en: "Our ancestors left us a rich cultural heritage.", label: "Culture" },
    { en: "She sang a beautiful song at the festival.", label: "Celebration" },
  ],
};
