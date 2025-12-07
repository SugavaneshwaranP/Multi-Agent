import argparse
from pipelines.sales_pipeline import SalesPipeline
from pipelines.resume_pipeline import ResumePipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['sales', 'resume'])
    args = parser.parse_args()

    if args.task == 'sales':
        pipeline = SalesPipeline('datasets/sales/output.xlsx', analysis_mode='advanced')
        result = pipeline.run()dayed
        print(result)
    elif args.task == 'resume':
        pipeline = ResumePipeline('datasets/resumes/sample_resume.txt')
        result = pipeline.run()
        print(result)

if __name__ == '__main__':
    main()
