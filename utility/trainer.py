import torch
from time import time, strftime, localtime
import utility.tools as tools
import utility.tester as tester
import os
import datetime


def training(model, args, dataset, device, logger):
    model.to(device)

    best_recall_epoch, best_recall_1, best_ndcg_1 = 0, 0, 0
    cnt = 0

    optim = torch.optim.Adam(model.parameters(), lr=float(args.learn_rate))
    logger.info(model)
    logger.info(optim)
    K = int(args.local_space_scale)
    logger.info("Local space scale K: %d" % K)

    for epoch in range(int(args.train_epoch)):
        start_time = time()

        model.train()

        local_spaces, num_local_spaces = tools.LocalSpace.sample_from_dataset(
            dataset, local_space_scale=K, device=device)

        for local_space_i, local_space in enumerate(local_spaces):

            loss_list = model(local_space)

            total_loss = 0.

            if local_space_i == 0:
                assert len(loss_list) > 1
                total_loss_list = [0.] * len(loss_list)

            for i in range(len(loss_list)):
                loss = loss_list[i]
                total_loss += loss
                total_loss_list[i] += loss.item()

            print('\t Local space %d/%d: loss = %.8f' % (local_space_i, num_local_spaces, total_loss), end='\r')
            optim.zero_grad()
            total_loss.backward()
            optim.step()
        end_time = time()


        loss = round(sum(total_loss_list) / num_local_spaces, 6)
        loss_strs = str(loss) + "=" + "+".join([str(round(i / num_local_spaces, 6)) for i in total_loss_list])
        print("\t Epoch: %4d| train time %.3f| train loss: %s" % (epoch + 1, end_time - start_time, loss_strs))
        logger.info("\t Epoch: %4d| train time %.3f| train loss: %s" % (epoch + 1, end_time - start_time, loss_strs))

        if epoch % int(args.test_frequency) == 0:
            if int(args.sparsity_test) == 0:
                model_results = tester.testing(model, args, dataset, device)
                cnt += 1
                if model_results['recall'][1] > best_recall_1:
                    cnt = 0
                    best_recall_epoch = epoch + 1
                    best_recall_1 = model_results['recall'][1]
                    best_ndcg_1 = model_results['ndcg'][1]
                    if not os.path.exists('pt'):
                        os.mkdir('pt')
                    if not os.path.exists('pt/'+model.model_name):
                        os.mkdir('pt/' + model.model_name)
                    if not os.path.exists('pt/'+model.model_name+'/'+args.dataset):
                        os.mkdir('pt/'+model.model_name+'/'+args.dataset)
                    torch.save({'user_embedding': model.user_embedding.state_dict(),
                                'item_embedding': model.item_embedding.state_dict(),
                               }, 'pt/'+ model.model_name+'/'+ args.dataset +'/user_item_embeddings.pt')
                if not os.path.exists('pt'):
                    os.mkdir('pt')
                if not os.path.exists('pt/'+model.model_name):
                    os.mkdir('pt/' + model.model_name)
                if not os.path.exists('pt/'+model.model_name+'/'+args.dataset):
                    os.mkdir('pt/'+model.model_name+'/'+args.dataset)
                torch.save({'user_embedding': model.user_embedding.state_dict(),
                            'item_embedding': model.item_embedding.state_dict(),
                           }, 'pt/'+ model.model_name+'/'+ args.dataset +'/user_item_now_embeddings.pt')
                print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']))
                logger.info(
                    "\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(
                        model_results['ndcg']))
                if cnt > int(args.early_stop):
                    break
            elif int(args.sparsity_test) == 1:
                result = tester.sparsity_test(dataset, args, model, device)
                if result[0]['ndcg'][1] > best_ndcg_1:
                    best_recall_epoch = epoch + 1
                    best_recall_1 = result[0]['recall'][1]
                    best_ndcg_1 = result[0]['ndcg'][1]
                print("\t level_1: recall:", result[0]['recall'], ',ndcg:',
                      result[0]['ndcg'])
                print("\t level_2: recall:", result[1]['recall'], ',ndcg:',
                      result[1]['ndcg'])
                print("\t level_3: recall:", result[2]['recall'], ',ndcg:',
                      result[2]['ndcg'])

                logger.info("\t level_1: recall:" + str(result[0]['recall']) + ',ndcg:' + str(result[0]['ndcg']))
                logger.info("\t level_2: recall:" + str(result[1]['recall']) + ',ndcg:' + str(result[1]['ndcg']))
                logger.info("\t level_3: recall:" + str(result[2]['recall']) + ',ndcg:' + str(result[2]['ndcg']))
            elif int(args.sparsity_test) == 2:
                result = tester.item_sparsity_test(dataset, args, model, device)
                if result[0]['ndcg'][1] > best_ndcg_1:
                    best_recall_epoch = epoch + 1
                    best_recall_1 = result[0]['recall'][1]
                    best_ndcg_1 = result[0]['ndcg'][1]
                group_names = ['tail-1 item', 'tail-2 item', 'mid item', 'head-2 item', 'head-1 item']
                for idx, group_name in enumerate(group_names):
                    print("\t %s: recall:" % group_name, result[idx]['recall'], ',ndcg:', result[idx]['ndcg'])
                    logger.info("\t %s: recall:" % group_name + str(result[idx]['recall']) + ',ndcg:' + str(result[idx]['ndcg']))

    print("\t Model training process completed.")
    print("\t best recall epoch:" + str(best_recall_epoch))
    print("\t best recall:" + str(best_recall_1) + "\t best ndcg:" + str(best_ndcg_1))

    logger.info("\t Model training process completed.")
    logger.info("\t best recall epoch:" + str(best_recall_epoch))
    logger.info("\t best recall:" + str(best_recall_1) + "\t best ndcg:" + str(best_ndcg_1))

    """
    save file to result folder
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    file_path = os.path.join(result_folder, f"{model.model_name}_results_{current_time}.txt")
    with open(file_path, 'w') as file:
        file.write("Model Training Results for {}:\n".format(model.model_name))
        file.write("Best Recall Epoch: " + str(best_recall_epoch) + "\n")
        file.write("Best Recall: " + str(best_recall_1) + "\n")
        file.write("Best NDCG: " + str(best_ndcg_1) + "\n")
        file.write("File created at: " + current_time + "\n")

    handlers = logger.handlers

    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
