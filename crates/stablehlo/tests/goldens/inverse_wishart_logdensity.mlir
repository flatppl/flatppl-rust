module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2x2xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.cholesky %arg0, lower = true : tensor<2x2xf32>
    %6 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %7 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %8 = stablehlo.compare EQ, %6, %7 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %9 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %10 = stablehlo.select %8, %5, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%10 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %13 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %14 = stablehlo.reduce(%12 init: %13) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %16 = stablehlo.reduce(%12 init: %15) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %17 = stablehlo.compare GT, %14, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %18 = stablehlo.compare LT, %16, %13 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %19 = stablehlo.select %17, %18, %17 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %20 = stablehlo.select %19, %2, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %21 = stablehlo.compare EQ, %4, %2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %22 = stablehlo.select %21, %20, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %23 = stablehlo.cholesky %arg2, lower = true : tensor<2x2xf32>
    %24 = stablehlo.log %12 : tensor<2xf32>
    %25 = stablehlo.reduce(%24 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %26 = stablehlo.constant dense<2.0> : tensor<f32>
    %27 = stablehlo.multiply %26, %25 : tensor<f32>
    %28 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %29 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %30 = stablehlo.compare EQ, %28, %29 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %31 = stablehlo.select %30, %23, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %32 = stablehlo.reduce(%31 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %33 = stablehlo.log %32 : tensor<2xf32>
    %34 = stablehlo.reduce(%33 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %35 = stablehlo.multiply %26, %34 : tensor<f32>
    %36 = "stablehlo.triangular_solve"(%23, %5) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %37 = stablehlo.multiply %36, %36 : tensor<2x2xf32>
    %38 = stablehlo.reduce(%37 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %39 = stablehlo.reduce(%38 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %40 = stablehlo.constant dense<0.5> : tensor<f32>
    %41 = stablehlo.multiply %40, %arg1 : tensor<f32>
    %42 = stablehlo.multiply %41, %27 : tensor<f32>
    %43 = stablehlo.constant dense<3.0> : tensor<f32>
    %44 = stablehlo.add %arg1, %43 : tensor<f32>
    %45 = stablehlo.constant dense<-0.5> : tensor<f32>
    %46 = stablehlo.multiply %45, %44 : tensor<f32>
    %47 = stablehlo.multiply %46, %35 : tensor<f32>
    %48 = stablehlo.multiply %45, %39 : tensor<f32>
    %49 = stablehlo.multiply %arg1, %26 : tensor<f32>
    %50 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %51 = stablehlo.multiply %49, %50 : tensor<f32>
    %52 = stablehlo.multiply %45, %51 : tensor<f32>
    %53 = stablehlo.constant dense<0.5723649429247001> : tensor<f32>
    %54 = stablehlo.add %41, %0 : tensor<f32>
    %55 = chlo.lgamma %54 : tensor<f32> -> tensor<f32>
    %56 = stablehlo.add %53, %55 : tensor<f32>
    %57 = stablehlo.add %41, %45 : tensor<f32>
    %58 = chlo.lgamma %57 : tensor<f32> -> tensor<f32>
    %59 = stablehlo.add %56, %58 : tensor<f32>
    %60 = stablehlo.negate %59 : tensor<f32>
    %61 = stablehlo.add %42, %47 : tensor<f32>
    %62 = stablehlo.add %61, %48 : tensor<f32>
    %63 = stablehlo.add %62, %52 : tensor<f32>
    %64 = stablehlo.add %63, %60 : tensor<f32>
    return %64, %22 : tensor<f32>, tensor<i32>
  }
}
