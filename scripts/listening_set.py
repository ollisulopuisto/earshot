"""Render a listening set: one folder per comparison, then the page.

    uv run python scripts/listening_set.py material/local/ears out/kuuntelu

Each comparison is one damage on one excerpt of one real voice, with the
untouched original first (the reference the page matches loudness to), the
damaged input second, and then every engine that is a guess at fixing it.
The guesses are deliberately several per damage: two wrong guesses heard
side by side say more than one number.

Written as 16-bit WAV to keep a published set small: eight seconds is
768 KB, and a set of ten comparisons fits in the 64 MB a published page may
carry. The page applies the loudness match itself, so no file is gained.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from earshot import degrade, engines, listen

SECONDS = 8.0

ORIGINAL = ("Alkuperäinen", "EARS-studioäänitys sellaisenaan: tähän verrataan")
DAMAGED = ("Vaurioitettu", "")

# (folder, speaker file, start s, recipe, title, what to listen for,
#  [(engine spec, label, note)])
PLAN = [
    ("01-puhdas", "p008 freeform 01.wav", 42.0, "clean",
     "Puhdas matala miesääni: saako mikään koskea tähän?",
     "Mitään ei ole rikottu. Paras otto on se, jota ei erota alkuperäisestä. "
     "Kuuntele rinta-ääntä (80–150 Hz) ja ylätaajuuksien ilmavuutta.",
     [("dehum:50", "Hurinanpoisto", "pitäisi olla bitilleen sama"),
      ("deplosive", "Plosiivirajoitin", "saa leikata vain pamauksia"),
      ("highpass:80", "Ylipäästö 80 Hz", "tavallinen tylppä ratkaisu"),
      ("deepfilternet:12", "DeepFilterNet, max 12 dB", "kohinanpoisto talutushihnassa"),
      ("router:lavasr", "LavaSR reitittimen kautta", "keksii vain tyhjälle kaistalle"),
      ("lavasr", "LavaSR aina päällä", "kaistanlaajennus kaikkeen")]),
    ("02-hurina", "p008 freeform 01.wav", 42.0, "hum",
     "Verkkohurina 50 Hz, ryömivä",
     "50 Hz ja viisi yläsävelä 30 dB puheen alla. Kuuntele tauoissa ja "
     "matalien vokaalien alla: jääkö hurinaa, ja ohentuuko ääni?",
     [("dehum:50", "Hurinanpoisto, 1 Hz lovi", "vähentää vain paikallaan pysyvän"),
      ("dehum:50@2", "Hurinanpoisto, 2 Hz lovi", "seuraa ryömintää väljemmin"),
      ("highpass:120", "Ylipäästö 120 Hz", "vie perustaajuuden ja puheen pohjan"),
      ("deepfilternet:12", "DeepFilterNet, max 12 dB", ""),
      ("deepfilternet", "DeepFilterNet rajoittamaton", "")]),
    ("03-surina", "p001 freeform 02.wav", 60.0, "buzz",
     "Virtalähteen surina: parittomat yläsävelet 4 kHz:iin asti",
     "Surina on hurinaa kirkkaampi ja ärsyttävämpi. Hurinanpoisto käsittelee "
     "vain puheen yli erottuvat yläsävelet, joten sitä verrataan kohinanpoistoon.",
     [("dehum:50", "Hurinanpoisto", ""),
      ("deepfilternet:12", "DeepFilterNet, max 12 dB", ""),
      ("deepfilternet", "DeepFilterNet rajoittamaton", ""),
      ("chain:dehum:50+deepfilternet:12", "Hurinanpoisto → DeepFilterNet 12", "")]),
    ("04-plosiivit", "p008 freeform 02.wav", 30.0, "plosive",
     "Plosiivipamaukset sanojen alussa",
     "Mallinnettuja p/b-pamauksia, 12 minuutissa. Kuuntele pamausten lisäksi "
     "ääntä niiden välissä: ylipäästö ohentaa kaiken, rajoittimen pitäisi "
     "ohentaa vain pamauksen.",
     [("deplosive", "Plosiivirajoitin < 150 Hz", "dynaaminen, puhujakohtainen"),
      ("deplosive:200", "Plosiivirajoitin < 200 Hz", ""),
      ("highpass:80", "Ylipäästö 80 Hz", ""),
      ("highpass:120", "Ylipäästö 120 Hz", "")]),
    ("05-huone", "p002 freeform 01.wav", 50.0, "room",
     "Kaikuva huone (RT60 0,6 s), korjatulla reseptillä",
     "Aiempi mittaus väitti DeepFilterNetin poistavan kaikuvasta puheesta "
     "60 dB, mutta resepti hiljensi samalla 28 dB. Tämä on ensimmäinen kuuntelu "
     "korjatulla reseptillä. Poistuuko puhe vai vain kaiku?",
     [("deepfilternet", "DeepFilterNet rajoittamaton", ""),
      ("deepfilternet:20", "DeepFilterNet, max 20 dB", ""),
      ("deepfilternet:12", "DeepFilterNet, max 12 dB", ""),
      ("lavasr", "LavaSR", ""),
      ("router:lavasr", "LavaSR reitittimen kautta", "")]),
    ("06-kohina", "p001 freeform 01.wav", 75.0, "hiss",
     "Laajakaistainen kohina 20 dB puheen alla",
     "Helppo tapaus, jossa mittarit ovat samaa mieltä. Kuuntele, kuulostaako "
     "kohinanpoiston jälkeinen ääni vielä ihmiseltä, ja auttaako LavaSR "
     "kohinanpoiston perässä vai pilaako se.",
     [("deepfilternet", "DeepFilterNet rajoittamaton", "PESQ +1,96 vanhassa mittauksessa"),
      ("deepfilternet:12", "DeepFilterNet, max 12 dB", ""),
      ("lavasr", "LavaSR", "ei ole kohinanpoistaja"),
      ("chain:deepfilternet:12+router:lavasr", "DFN 12 → reititin(LavaSR)", "")]),
    ("07-alusta", "p002 freeform 02.wav", 40.0, "platform-upload",
     "Etätallennusalusta: tauot digitaalista nollaa, kaista 15 kHz:iin",
     "Oikean alustamateriaalin pääasiallinen vika on portitus. LavaSR lisäsi "
     "vanhassa mittauksessa lattiatasoon 34,5 dB. Kuuntele taukojen reunoja: "
     "sihiseekö vai napsuuko?",
     [("lavasr", "LavaSR", ""),
      ("keepzero:lavasr", "LavaSR, hiljaisuus palautettu", "nollat takaisin nolliksi"),
      ("router:lavasr", "LavaSR reitittimen kautta", ""),
      ("deepfilternet:12", "DeepFilterNet, max 12 dB", "")]),
    ("08-puhelin", "p001 freeform 01.wav", 75.0, "narrowband-voip",
     "Huono puhelu: 300–3400 Hz, klippaus, pakettihäviö, automaattinen taso",
     "Tässä tapauksessa keksiminen on ainoa keino. Vertaa, kuulostaako "
     "keksitty yläpää samalta ihmiseltä kuin alkuperäinen.",
     [("lavasr", "LavaSR", ""),
      ("router:lavasr", "LavaSR reitittimen kautta", ""),
      ("chain:declip+lavasr", "Declip → LavaSR", ""),
      ("chain:deepfilternet:12+lavasr", "DFN 12 → LavaSR", "")]),
    ("09-klippaus", "p002 freeform 01.wav", 50.0, "clipped",
     "Pelkkä kova klippaus 6 dB huipun alla",
     "Klippausta on oikeassa materiaalissa miljoonasosien verran, joten tämä "
     "on varotoimi eikä pääasia. Kuuluuko rätinä, ja poistaako paikallinen "
     "piirto sen?",
     [("declip", "Declip", "kuutiollinen piirto tasanteiden yli")]),
    ("10-opus", "p008 freeform 02.wav", 30.0, "opus-16k",
     "Opus 16 kbit/s, oikea koodekki",
     "Mitattuna mikään ei auttanut: kaistaa ei puutu, vain yksityiskohtia. "
     "Kuuntele, vahvistaako korva mittauksen.",
     [("lavasr", "LavaSR", ""),
      ("router:lavasr", "LavaSR reitittimen kautta", ""),
      ("deepfilternet:12", "DeepFilterNet, max 12 dB", "")]),
]


def main(source: Path, target: Path) -> None:
    cache: dict[str, engines.Loaded] = {}
    target.mkdir(parents=True, exist_ok=True)
    for folder, speaker_file, start, recipe_name, title, listen_for, takes in PLAN:
        started = time.time()
        clean, rate = sf.read(source / speaker_file, dtype="float32",
                              start=int(start * 48000), frames=int(SECONDS * 48000))
        recipe = degrade.by_name(recipe_name)
        damaged = recipe.apply(clean, rate)
        out = target / folder
        out.mkdir(parents=True, exist_ok=True)
        labels = {}

        def write(index: int, name: str, audio: np.ndarray, label: str, note: str):
            filename = f"{index:02d}-{name}.wav"
            sf.write(out / filename, np.clip(audio, -1, 1), rate, subtype="PCM_16")
            labels[filename] = {"label": label, "note": note}

        write(0, "alkuperainen", clean, *ORIGINAL)
        index = 1
        if recipe_name != "clean":
            write(1, "vaurio", damaged, DAMAGED[0], recipe.describe)
            index = 2
        for spec, label, note in takes:
            if spec not in cache:
                cache[spec] = engines.load(spec)
            engine = cache[spec].engine
            restored = engines.check_contract(damaged, engine.process(damaged, rate), engine.name)
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in engine.name)
            write(index, safe, restored, label, note)
            index += 1
        speaker = speaker_file.split()[0]
        (out / "about.json").write_text(json.dumps({
            "title": title,
            "listen_for": f"{listen_for} Puhuja {speaker}, {start:g}–{start + SECONDS:g} s.",
            "takes": labels,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{folder}: {index} takes in {time.time() - started:.1f} s", flush=True)

    intro = INTRO
    listen.build(target, title="Earshot-kuuntelu", intro=intro)
    print(f"wrote {target / 'index.html'}")


INTRO = """
<p>Kymmenen vertailua, joissa jokainen kokeilu on arvaus. Materiaali on
EARS-korpuksen studioäänitystä (kolme englanninkielistä puhujaa, 48 kHz),
johon vauriot on tehty tarkoituksella. Siksi alkuperäinen on aina tallessa
ensimmäisenä.</p>
<p>Otot soivat samanaikaisesti, ja valinta vaihtaa vain kuuluvan oton, joten
vertailu tapahtuu kesken tavun. <b>Tasoitus</b> säätää kaikki otot
alkuperäisen puhetasolle, jotta kovempi ei voita vain siksi, että se on
kovempi. <b>Sokko</b> sekoittaa ottojen järjestyksen ja piilottaa nimet.</p>
<p>EARS on lisensoitu CC BY-NC 4.0 -ehdoin. Aineisto on tarkoitettu vain
arviointiin eikä sitä ole viety repoon.</p>
"""


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
