from tcgdexsdk import TCGdex, Language
import json

sdk = TCGdex()

sets = ["svp", "sv01", "sv02", "sv03", "sv03.5", "sv04", "sv04.5", "sv05", "sv06", "sv06.5", "sv07", "sv08", "sv08.5", "sv09", 
"sv10", "sv10.5b","sv10.5w", "me01", "me02", "me02.5", "me03", "me04", "me05",]

for currSet in sets:
    set_data = sdk.set.getSync(currSet)
                                            
    all_cards = []
    for card in set_data.cards:
        all_cards.append({
            "name": card.name,
            "localId": card.localId,
            "set_id": set_data.id,
            "image_url": card.get_image_url(quality="high", extension="png"),
        })

    with open(f"data/sets/{currSet}.json", "w") as f:
        json.dump(all_cards, f, indent=1)
