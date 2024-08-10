import soundDownload as SD
import soundAnalysis as SA
#
key = 'E1OXAiFtTXP2uPqoVcZxsaLsjD9EaNEBmX6eqvwQ'
#
# SD.downloadSoundsFreesound(queryText='sax', API_Key=key, outputDir='.\\testDownload', topNResults=20, duration=(0,8),
#                            tag='neumann-u87')
# SD.downloadSoundsFreesound(queryText='guitar', API_Key=key, outputDir='.\\testDownload', topNResults=20, duration=(0,8),
#                            tag='multisample')
# SD.downloadSoundsFreesound(queryText='hihat', API_Key=key, outputDir='.\\testDownload', topNResults=20, duration=(0,8),
#                            tag='1-shot')

min_value = 99999  # Changed variable name to avoid conflict with the built-in min function
index1 = index2 = None  # Initialize indices

for i in range(0, 17):
    for j in range(0, 17):
        if i != j:
            b = SA.clusterSounds('.\\testDownload\\', nCluster=3, descInput=[i, j])
            a = int(b)  # Convert the desired substring to an integer
            if a < min_value:
                min_value = a
                index1 = i
                index2 = j


if index1 is not None and index2 is not None:
    print('min is:', min_value, 'for indices', index1, index2)
else:
    print('No valid clusters found.')

SD.downloadSoundsFreesound(queryText='bass', API_Key=key, outputDir='.\\testDownload', topNResults=1, duration=(0,8))