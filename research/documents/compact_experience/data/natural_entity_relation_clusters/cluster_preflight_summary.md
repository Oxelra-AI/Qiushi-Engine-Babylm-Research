# natural cluster preflight review natural entity-relation cluster preflight

Reads only official Strict-Small training text. No official AoA/CDI words, child curves, AoA outputs, SuperGLUE labels, or downstream evaluation results are read or used.

Sentences with anchors: 158523
Clusters found: 21772
Clusters with held-out same-block sentence: 3435
2% prefix: 2355 clusters / 199971 words
4% prefix: 5140 clusters / 399971 words

## Top source counts
- simple_wiki.train.txt: 13266
- gutenberg.train.txt: 4273
- bnc_spoken.train.txt: 1738
- open_subtitles.train.txt: 1296
- childes.train.txt: 1185
- switchboard.train.txt: 14

## First samples
- `5326b0bd1f51d4ac` gutenberg.train.txt anchor=cap:kalli_lal words=127 heldout=True
  - Of medium height and active figure, the stranger watched the approach of the visitors, but, unlike Kalli Lal, he was attired in ordinary English style and wore a small black moustache.
  - Accompanied by Kalli Lal the visitors ascended the stairs, and on entering the 'palace' suddenly faced a double line of immovable figures, apparently acting as a 'guard of honour,' the smooth velvet-like brown bodies being nude to the waist.
  - For one moment Jack Clewlin caught the searching glance of Kalli Lal fixed on his protectors, and he thought that an approving smile crossed the Malay's lips; but the next moment a bamboo screen of native make was drawn aside, and the party entered a chamber of considerable dimensions, and almost wholly furnished after the European manner.
- `d196ca70c45f0b15` gutenberg.train.txt anchor=cap:roger_williams words=135 heldout=True
  - The disfranchisement of Catholics was contrary to the spirit of the Rhode Island charter and to the views of Roger Williams, who certainly understood the rational grounds for religious toleration better than any other man of his time, save perhaps Milton and Vane.
  - Roger Williams not only proclaimed such doctrines, but he lived up to them.
  - Massachusetts in fury threatened to cut off the trade of the weaker colony, but nothing could intimidate Williams into what he termed "exercising a civil power over men's consciences." Among the public men of the seventeenth century Roger Williams deserves a preëminent place; he was the first to conceive thoroughly and carry out consistently, in the face of strong opposition, a theory of religious liberty broad enough to win assent and approval from advanced thinkers of the present day.
- `914fbae842edad71` gutenberg.train.txt anchor=cap:rhode_island words=128 heldout=True
  - Even the laws of Rhode Island, as first printed, early in the eighteenth century, expressly prohibit Roman Catholics from voting.
  - The disfranchisement of Catholics was contrary to the spirit of the Rhode Island charter and to the views of Roger Williams, who certainly understood the rational grounds for religious toleration better than any other man of his time, save perhaps Milton and Vane.
  - He never took pains to conceal his dislike of Quaker doctrines; in his seventy-third year he once rowed himself in a boat the whole length of Narragansett Bay, in order to conduct a dispute against three valiant Quaker champions; yet, in spite of vehement pressure from the neighbouring colonies, he resolutely refused to allow the civil power of Rhode Island to be used against Quakers.
- `ec08e127465f2d94` gutenberg.train.txt anchor=cap:ocean_glory words=134 heldout=True
  - Owing to some mistake about the nationality of the 'black-birding' schooner, the 'Ocean Glory' was detained till nearly sundown, but when the chief officer sang out to man the windlass all hands rushed to the levers.
  - As the anchor was broken out the 'Ocean Glory' canted her head seaward, and under all sail, and with bunting flowing bravely in the breeze, away toward the offing she glided with ever-increasing movement.
  - The 'Ocean Glory' continued her voyage, crossed the equator in good style, and after a delay of only one day she struck the first of the south-east trade winds, and in one long close-hauled board stood away about south-west-by-south, still keeping a sharp watch for the slightest sign of her opponent, and making rapid progress toward the bleak and stormy latitudes of Cape Horn.
- `401642bed626f0db` gutenberg.train.txt anchor=cap:tiny_tim words=135 heldout=True
  - Cratchit made the gravy (ready beforehand in a little saucepan) hissing hot; Master Peter mashed the potatoes with incredible vigour; Miss Belinda sweetened up the apple sauce; Martha dusted the hot plates; Bob took Tiny Tim beside him in a tiny corner at the table; the two young Cratchits set chairs for everybody, not forgetting themselves, and, mounting guard upon their posts, crammed spoons into their mouths, lest they should shriek for goose before their turn came to be helped.
  - Tiny Tim drank it last of all, but he didn't care twopence for it.
  - All this time the chestnuts and the jug went round and round; and by-and-by they had a song, about a lost child travelling in the snow, from Tiny Tim, who had a plaintive little voice, and sang it very well indeed.
