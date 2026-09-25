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
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from earshot import degrade, engines, listen

SECONDS = 8.0

ORIGINAL = ("Alkuperäinen", "EARS-studioäänitys sellaisenaan: tähän verrataan")
DAMAGED = ("Vaurioitettu", "")

# The platform recipe's gate sits 18 dB under the speech, calibrated on
# podcast studio tracks whose room tone is about 22 dB down. EARS pauses sit
# only about 18 dB under its speech, so on EARS freeform speech that gate
# closed for 0.0 per cent of the time and `platform-upload` was a plain
# 15 kHz low-pass. For listening, the gate is set where it gives this read
# passage 25 per cent exact zero, inside the 8-58 per cent measured on real
# platform files.
PLATFORM_EARS = degrade.Damage(
    "platform-upload",
    "gated silence (-14 dB gate, 25 % exact zero) and a 15 kHz ceiling",
    ((degrade.gate, {"threshold_db": -14.0}),
     (degrade.band_limit, {"low": 0.0, "high": 15000.0})),
)

# (folder, speaker file, start s, recipe, title, what to listen for,
#  [(engine spec, label, note)])
EARS = [
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
    ("07-alusta", "../ears-sent/p008/rainbow_03_regular.wav", 0.0, PLATFORM_EARS,
     "Etätallennusalusta: tauot digitaalista nollaa, kaista 15 kHz:iin",
     "Oikean alustamateriaalin pääasiallinen vika on portitus: tässä 29,7 % "
     "näytteistä on nollaa. Tauoissa LavaSR jättää −74,8 dBFS, sama "
     "hiljaisuus palautettuna −96,1, reititin −117,2 ja DeepFilterNet −90,2. "
     "Kuuntele taukojen reunoja: sihiseekö vai napsuuko?",
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


@dataclass(frozen=True)
class Plan:
    comparisons: list
    original: tuple[str, str]
    intro: str
    title: str


EARS_INTRO = """
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

PODCAST = [
    ("01-puhdas", "nyman a.wav", 15.6, "clean", "Puhdas lähimikrofoni",
     "Alkuperäinen on jo puhdas. Kuuntele, säilyvätkö matala ääni ja ilmavuus.",
     [("dehum:50", "Hurinanpoisto", ""), ("notch:50", "50 Hz lovi", ""),
      ("deplosive", "Plosiivirajoitin", ""), ("highpass:80", "Ylipäästö 80 Hz", ""),
      ("deepfilternet:12", "DeepFilterNet 12 dB", "")]),
    ("02-lovi-wancke", "wancke b.wav", 20.0, "clean", "Lovi etävieraan äänessä",
     "Tässä lovi heikensi PESQ-tulosta 0,56. Kuuntele äänen ohentumista.",
     [("notch:50", "50 Hz lovi", ""), ("dehum:50", "Hurinanpoisto", "")]),
    ("03-huone-nyman", "nyman b.wav", 4.0, "room", "Huonekaiku, Nyman",
     "Poistuuko kaiku puheen mukana?", [("deepfilternet", "DeepFilterNet", ""),
      ("deepfilternet:20", "DeepFilterNet 20 dB", ""), ("deepfilternet:12", "DeepFilterNet 12 dB", "")]),
    ("04-huone-wancke", "wancke a.wav", 51.6, "room", "Huonekaiku, Wancke",
     "Poistuuko kaiku puheen mukana?", [("deepfilternet", "DeepFilterNet", ""),
      ("deepfilternet:20", "DeepFilterNet 20 dB", ""), ("deepfilternet:12", "DeepFilterNet 12 dB", "")]),
    ("05-hurina", "nyman a.wav", 23.6, "hum", "Verkkohurina",
     "Kuuntele taukoja ja matalia vokaaleja.", [("dehum:50", "Hurinanpoisto", ""),
      ("notch:50", "50 Hz lovi", ""), ("highpass:80", "Ylipäästö 80 Hz", ""),
      ("deepfilternet:12", "DeepFilterNet 12 dB", "")]),
    ("06-maadoitus", "wancke b.wav", 12.0, "ground-loop", "Maadoitussilmukka",
     "Kuuntele hurinan lisäksi puheen rungon säilymistä.", [("notch:50", "50 Hz lovi", ""),
      ("dehum:50", "Hurinanpoisto", ""), ("deepfilternet:12", "DeepFilterNet 12 dB", "")]),
    ("07-kohina", "nyman b.wav", 50.8, "hiss", "Laajakaistainen kohina",
     "Kuuntele, säilyykö puhe luonnollisena.", [("deepfilternet", "DeepFilterNet", ""),
      ("deepfilternet:20", "DeepFilterNet 20 dB", ""), ("deepfilternet:12", "DeepFilterNet 12 dB", "")]),
    ("08-plosiivit", "wancke a.wav", 1.2, "plosive", "Plosiivit",
     "Kuuntele pamauksia ja niiden välistä ääntä.", [("deplosive", "Plosiivirajoitin", ""),
      ("highpass:80", "Ylipäästö 80 Hz", "")]),
    ("09-klippaus", "nyman a.wav", 33.6, "clipped", "Klippaus",
     "Kuuluuko särö, ja auttaako de-clip?", [("declip", "Declip", "")]),
]

PODCAST_INTRO = """
<p>Vertailut on tehty pp53-podcastin nyman- ja wancke-äänitteistä.
Tämä on yksityistä aineistoa: kuuntelusettiä ei saa julkaista eikä jakaa.</p>
<p>Otot soivat samanaikaisesti. <b>Tasoitus</b> säätää niiden tason vertailua
varten ja <b>Sokko</b> piilottaa nimet.</p>
"""

SETS = {
    "ears": Plan(EARS, ("Alkuperäinen", "EARS-studioäänitys sellaisenaan: tähän verrataan"), EARS_INTRO, "Earshot-kuuntelu"),
    "podcast": Plan(PODCAST, ("Alkuperäinen", "pp53-podcastin puhdas äänite"), PODCAST_INTRO, "Podcast-kuuntelu (yksityinen)"),
}


def main(source: Path, target: Path, plan: Plan, only: set[str] | None = None) -> None:
    cache: dict[str, engines.Loaded] = {}
    target.mkdir(parents=True, exist_ok=True)
    for folder, speaker_file, start, recipe_name, title, listen_for, takes in plan.comparisons:
        if only and folder not in only:
            continue
        started = time.time()
        clean, rate = sf.read(source / speaker_file, dtype="float32",
                              start=int(start * 48000), frames=int(SECONDS * 48000))
        recipe = (recipe_name if isinstance(recipe_name, degrade.Damage)
                  else degrade.by_name(recipe_name))
        recipe_name = recipe.name
        damaged = recipe.apply(clean, rate)
        out = target / folder
        out.mkdir(parents=True, exist_ok=True)
        labels = {}

        def write(index: int, name: str, audio: np.ndarray, label: str, note: str):
            filename = f"{index:02d}-{name}.wav"
            sf.write(out / filename, np.clip(audio, -1, 1), rate, subtype="PCM_16")
            labels[filename] = {"label": label, "note": note}

        write(0, "alkuperainen", clean, *plan.original)
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
        speaker = Path(speaker_file).parent.name or speaker_file.split()[0]
        (out / "about.json").write_text(json.dumps({
            "title": title,
            "listen_for": f"{listen_for} Puhuja {speaker}, {start:g}–{start + SECONDS:g} s.",
            "takes": labels,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{folder}: {index} takes in {time.time() - started:.1f} s", flush=True)

    listen.build(target, title=plan.title, intro=plan.intro)
    print(f"wrote {target / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=SETS, default="ears")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("only", nargs="*")
    args = parser.parse_args()
    main(args.source, args.target, SETS[args.set], set(args.only) or None)
