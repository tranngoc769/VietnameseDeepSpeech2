for i in range(3,5):
      file_object = open('epoch.log', 'a')
      file_object.write('Training Summary Epoch: [{0}]\t'
                  'Time taken (s): {epoch_time:.0f}\t'
                  'Average Loss {loss:.3f}\n'.format(i+1, epoch_time=5.12, loss=4552))
      file_object.close()
      file_object = open('epoch.log', 'a')
      file_object.write('Training Summary Epoch: [{0}]\t'
                  'Time taken (s): {epoch_time:.0f}\t'
                  'Average Loss {loss:.3f}\n'.format(i+1, epoch_time=5.12, loss=4552))
      file_object.close()